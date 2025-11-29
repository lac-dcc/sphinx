#pragma once

#include "proto_cpp/program_graph.pb.h"

#include "llvm/ADT/DenseMap.h"

#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/Value.h"
#include "mlir/IR/AsmState.h"

#include "program_graph_builder.h"

using programl::Node;
using programl::Function;
using programl::Module;

using FunctionEntryExits = std::pair<Node *, std::vector<Node *>>;
using BlockEntryExit = std::pair<Node *, Node *>;

class MLIRToProGraMLBuilder : public programl::graph::ProgramGraphBuilder  {
public:
    MLIRToProGraMLBuilder() = default;

    ~MLIRToProGraMLBuilder() = default;

    [[nodiscard]] programl::ProgramGraph Build(mlir::ModuleOp module);

protected:
    [[nodiscard]] programl::ProgramGraph VisitModule(mlir::ModuleOp module);

    [[nodiscard]] FunctionEntryExits VisitFunction(mlir::Operation *op, const Function *functionMessage);

    [[nodiscard]] BlockEntryExit VisitBlock(
        mlir::Block &block, const Function *functionMessage,
        llvm::DenseMap<const mlir::Block *, BlockEntryExit> &blocks
    );

private:
    [[nodiscard]] Node *GetOrCreateOpNode(mlir::Operation *op, const Function *function);

    [[nodiscard]] Node *GetOrCreateValueNode(const mlir::Value &val, const Function *function);

    [[nodiscard]] Node *GetOrCreateConstNode(const mlir::NamedAttribute &attr);

    void CreateCallEdges(const Node *source, const FunctionEntryExits &target);

    std::optional<mlir::AsmState> asmState_;

    llvm::DenseMap<mlir::Operation *, Node *> op_nodes_;
    llvm::DenseMap<mlir::Value, Node *> value_nodes_;
    llvm::DenseMap<mlir::Attribute, Node *> const_nodes_;
    llvm::DenseMap<Node *, mlir::Operation *> call_sites_;
    llvm::DenseMap<Node *, int32_t> control_edge_pos;
};