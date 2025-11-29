#include "MLIRToProGraMLBuilder.h"

#include "llvm/Support/raw_ostream.h"
#include "mlir/IR/OpDefinition.h"
#include "mlir/Interfaces/FunctionInterfaces.h"
#include <mlir/Interfaces/ControlFlowInterfaces.h>
#include <mlir/Interfaces/LoopLikeInterface.h>

#include "proto_cpp/features.pb.h"

#include "create_features.h"

#include "mlir/Analysis/Liveness.h"


namespace {
    std::string TypeToString(const mlir::Type &type) {
        std::string s;
        llvm::raw_string_ostream os {s};
        type.print(os);
        return s;
    }
}


BlockEntryExit MLIRToProGraMLBuilder::VisitBlock(
    mlir::Block &block, const Function *functionMessage,
    llvm::DenseMap<const mlir::Block *, BlockEntryExit> &blocks
) {
    if (block.empty()) {
        block.getParentOp()->emitError("Block contains no operations");
        return {nullptr, nullptr};
    }

    Node *firstNode {nullptr}; // Entry point
    Node *lastNode {nullptr}; // Exit point (Terminator)

    for (mlir::Operation &op : block) {
        Node *opNode {GetOrCreateOpNode(&op, functionMessage)};
        const int32_t operandCount {static_cast<int32_t>(op.getOperands().size())};

        // Operand values (use) edges: VALUE -> OPERATION with operand index.
        for (auto it : llvm::enumerate(op.getOperands())) {
            const Node *useValueNode {GetOrCreateValueNode(it.value(), functionMessage)};
            (void) AddDataEdge(static_cast<int32_t>(it.index()), useValueNode, opNode);
        }

        // Operation attributes (use) edges: CONSTANT -> OPERATION with attribute index.
        for (auto it : llvm::enumerate(op.getAttrs())) {
            const Node *attrNode {GetOrCreateConstNode(it.value())};
            (void) AddDataEdge(operandCount + static_cast<int32_t>(it.index()), attrNode, opNode);
        }

        // Result values (def) edges: OPERATION -> VALUE with result index.
        for (auto it : llvm::enumerate(op.getResults())) {
            const Node *defValueNode {GetOrCreateValueNode(it.value(), functionMessage)};
            (void) AddDataEdge(static_cast<int32_t>(it.index()), opNode, defValueNode);
        }

        // Control flow edges for sequential ops inside a block.
        if (lastNode)
            (void) AddControlEdge(control_edge_pos[lastNode]++, lastNode, opNode);

        // If the operation is a function call (CallOpInterface trait), record the call site
        // Will be used later to create call edges
        if (auto callOp = llvm::dyn_cast<mlir::CallOpInterface>(op)) {
            if (mlir::Operation *calleeOp {callOp.resolveCallable()})
                call_sites_.insert({opNode, calleeOp});
            else
                op.emitError("Call site has no defined callee");
        }

        if (!firstNode)
            firstNode = opNode;
        lastNode = opNode;

        if (op.getNumRegions() > 0) {
            const bool isLoop {llvm::isa<mlir::LoopLikeOpInterface>(op)};

            const Node *nextOpNode {nullptr};
            if (auto *nextOp {op.getNextNode()})
                nextOpNode = GetOrCreateOpNode(nextOp, functionMessage);

            for (mlir::Region &region : op.getRegions()) {
                if (region.empty())
                    continue;

                for (mlir::Block &nestedBlock : region) {
                    BlockEntryExit nestedEntryExit {VisitBlock(nestedBlock, functionMessage, blocks)};
                    if (!nestedEntryExit.first || !nestedEntryExit.second) {
                        nestedBlock.getParentOp()->emitWarning("Skipping block with empty or invalid entry/exit nodes");
                        continue;
                    }

                    blocks.insert({&nestedBlock, nestedEntryExit});

                    const auto successors {nestedBlock.getSuccessors()};
                    const bool hasInternalSuccessor {
                        llvm::any_of(successors,
                            [&](const mlir::Block *successor) {
                                return successor->getParent() == &region;
                            }
                        )
                    };

                    // If the terminator has no explicit successors within its own region,
                    // it's a structured exit. Control flows back to the parent op's
                    // continuation point (nextOpNode).
                    if (!hasInternalSuccessor && nextOpNode) {
                        (void) AddControlEdge(control_edge_pos[nestedEntryExit.second]++, nestedEntryExit.second, nextOpNode);

                        // If the op is a loop, also connect this terminator back to the original op
                        if (isLoop)
                            (void) AddControlEdge(control_edge_pos[nestedEntryExit.second]++, nestedEntryExit.second, opNode);
                    }
                }

                mlir::Block &entryBlock {region.front()};
                auto it {blocks.find(&entryBlock)};
                if (it != blocks.end() && it->second.first) {
                    // Add a control edge from the op to the region entry (entry edge)
                    (void) AddControlEdge(control_edge_pos[opNode]++, opNode, it->second.first);
                }
            }
        }
    }

    return {firstNode, lastNode};
}


FunctionEntryExits MLIRToProGraMLBuilder::VisitFunction(mlir::Operation *op, const Function *functionMessage) {
    llvm::DenseMap<const mlir::Block *, BlockEntryExit> blocks {};
    FunctionEntryExits functionEntryExits {};

    auto funcInterface {llvm::dyn_cast<mlir::FunctionOpInterface>(op)};
    if (!funcInterface) {
        op->emitError("VisitFunction called on non-function op");
        return {};
    }

    if (funcInterface.isDeclaration()) {
        Node *node {AddInstruction("; undefined function", functionMessage)};
        programl::graph::AddScalarFeature(node, "full_text", "");
        functionEntryExits.first = node;
        functionEntryExits.second.push_back(node);
        return functionEntryExits;
    }

    mlir::Region &funcBody {funcInterface.getFunctionBody()};
    for (mlir::Block &block : funcBody) {
        BlockEntryExit entryExit {VisitBlock(block, functionMessage, blocks)};
        if (!entryExit.first || !entryExit.second) {
            block.getParentOp()->emitWarning("Skipping block with empty or invalid entry/exit nodes");
            continue;
        }
        blocks.insert({&block, entryExit});
    }

    if (blocks.empty()) {
        funcInterface.emitError() << "Function contains no blocks";
        return {};
    }

    mlir::Block &firstBlock {funcBody.front()};
    functionEntryExits.first = blocks.at(&firstBlock).first;

    llvm::DenseSet<const mlir::Block *> visited {&firstBlock};
    std::deque<mlir::Block *> q {&firstBlock};

    while (!q.empty()) {
        mlir::Block *current {q.front()};
        q.pop_front();

        Node *currentExit {blocks.at(current).second};

        // For each current -> successor pair, construct a control edge from the
        // last operation in the current block to the first operation in
        // the successor block.
        for (mlir::Block *successor: current->getSuccessors()) {
            if (auto it {blocks.find(successor)}; it != blocks.end()) {
                const Node *successorEntry {it->second.first};
                (void) AddControlEdge(control_edge_pos[currentExit]++, currentExit, successorEntry);

                if (visited.insert(successor).second)
                    q.push_back(successor);
            }
        }

        // A block's terminator is a function exit/return point if it has
        // the 'ReturnLike' trait.
        mlir::Operation *terminator {current->getTerminator()};
        if (terminator->hasTrait<mlir::OpTrait::ReturnLike>())
            functionEntryExits.second.push_back(currentExit);
    }

    return functionEntryExits;
}


programl::ProgramGraph MLIRToProGraMLBuilder::VisitModule(mlir::ModuleOp module) {
    const string moduleName {module.getName() ? module.getName().value() : "mlir_module"};
    const Module *moduleMessage {AddModule(moduleName)};

    std::vector<mlir::Operation *> functionOps {};

    // Iterate over all function definition operations in the module.
    // Function definitions will then be visited one by one.
    module->walk([&](mlir::FunctionOpInterface funcOp) {
        functionOps.push_back(funcOp.getOperation());
    });

    llvm::DenseMap<mlir::Operation *, FunctionEntryExits> functions {};

    for (mlir::Operation *funcOp : functionOps) {
        std::string funcDecl {funcOp->getName().getStringRef().str()};

        if (auto symbolAttr {funcOp->getAttrOfType<mlir::StringAttr>("sym_name")})
            funcDecl += " @" + symbolAttr.getValue().str();

        const Function *functionMessage {AddFunction(funcDecl, moduleMessage)};
        FunctionEntryExits functionEntryExits {VisitFunction(funcOp, functionMessage)};
        functions.insert({funcOp, functionEntryExits});
    }

    // Add call edges to and from the root node.
    for (const auto& function: functions)
        CreateCallEdges(GetRootNode(), function.second);

    // Add call edges to and from call sites.
    for (const auto &callSite : call_sites_) {
        if (auto it {functions.find(callSite.second)}; it != functions.end()) {
            const auto &calledFunctionEntryExits {it->second};
            CreateCallEdges(callSite.first, calledFunctionEntryExits);
        }
    }

    // ############### Liveness Testing ###############

    // We use the 'module' op as the top-level operation
    llvm::outs() << "--- Starting Liveness Analysis ---\n";
    mlir::Operation *topLevelOp = module.getOperation();

    llvm::outs() << "Running mlir::Liveness on the top-level operation...\n";
    mlir::Liveness liveness(topLevelOp);
    llvm::outs() << "Liveness analysis complete. Printing results:\n";

    // Print all liveness information
    liveness.print(llvm::outs());

    llvm::outs() << "--- Liveness Analysis Finished ---\n";


    // ############### Liveness Testing ###############

    return ProgramGraphBuilder::Build().ValueOrDie();
}


programl::ProgramGraph MLIRToProGraMLBuilder::Build(mlir::ModuleOp module) {
    asmState_.emplace(module);
    return VisitModule(module);
}


Node *MLIRToProGraMLBuilder::GetOrCreateOpNode(mlir::Operation *op, const Function *function) {
    const auto it {op_nodes_.find(op)};
    if (it != op_nodes_.end())
        return it->second;

    const std::string text {op->getName().getStringRef().str()};
    Node *node {AddInstruction(text, function)};

    std::string opStr;
    llvm::raw_string_ostream os {opStr};
    op->print(os, mlir::OpPrintingFlags().skipRegions());
    programl::graph::AddScalarFeature(node, "full_text", opStr);

    op_nodes_[op] = node;
    return node;
}


Node *MLIRToProGraMLBuilder::GetOrCreateValueNode(const mlir::Value &val, const Function *function) {
    const auto it {value_nodes_.find(val)};
    if (it != value_nodes_.end())
        return it->second;

    Node *node {AddVariable(TypeToString(val.getType()), function)};

    std::string valueStr;
    llvm::raw_string_ostream os {valueStr};
    val.printAsOperand(os, *asmState_);
    os << " : ";
    val.getType().print(os);
    programl::graph::AddScalarFeature(node, "full_text", valueStr);

    value_nodes_[val] = node;
    return node;
}


Node *MLIRToProGraMLBuilder::GetOrCreateConstNode(const mlir::NamedAttribute &attr) {
    const auto it {const_nodes_.find(attr.getValue())};
    if (it != const_nodes_.end())
        return it->second;

    string attrIdentifier {};
    if (const auto typed {mlir::dyn_cast<mlir::TypedAttr>(attr.getValue())}) {
        const mlir::Type type {typed.getType()};
        attrIdentifier = TypeToString(type);
    } else {
        llvm::raw_string_ostream os {attrIdentifier};
        os << attr.getName().str();
    }

    Node *node {AddConstant(attrIdentifier)};

    std::string attrStr;
    llvm::raw_string_ostream os {attrStr};
    attr.getValue().print(os);
    programl::graph::AddScalarFeature(node, "full_text", attrStr);

    const_nodes_[attr.getValue()] = node;
    return node;
}


void MLIRToProGraMLBuilder::CreateCallEdges(const Node *source, const FunctionEntryExits &target) {
    (void) AddCallEdge(source, target.first);
    for (const auto &exitNode: target.second)
        (void) AddCallEdge(exitNode, source);
}