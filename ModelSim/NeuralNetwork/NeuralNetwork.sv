module NeuralNetwork #(
    parameter dataWidth = 16, numInputs = 784, numOutputs = 10, L0neurons = 16, L1neurons = 10
) (
    input   logic                               clk,
    input   logic                               reset, 
    input   logic   [dataWidth*numInputs-1:0]   NNin,
    input   logic                               NNvalid,

    output  logic   [dataWidth*numOutputs-1:0]  NNout,
    output  logic                               NNoutValid,
    output  logic   [3:0]                       maxIndex,
    output  logic   [dataWidth-1:0]             maxValue,
    output  logic                               maxValid

);

logic [dataWidth*L0neurons-1:0] layer0Out;
logic                           layer0OutValid;
logic [dataWidth*L1neurons-1:0] layer1Out;
logic                           layer1OutValid;


layer0 #(
    .layerNumber(0), 
    .dataWidth(16), 
    .numInputs(784), 
    .numNeurons(16)
) layer0_inst (
    // Inputs
    .clk(clk),
    .reset(reset),
    .layerIn(NNin),
    .layerValid(NNvalid),
    // Outputs
    .layerOut(layer0Out),
    .layerOutValid(layer0OutValid)
);

layer1 #(
    .layerNumber(1), 
    .dataWidth(16), 
    .numInputs(16), 
    .numNeurons(10)
) layer1_inst (
    // Inputs
    .clk(clk),
    .reset(reset),
    .layerIn(layer0Out),
    .layerValid(layer0OutValid),
    // Outputs
    .layerOut(NNout),
    .layerOutValid(NNoutValid)
);

hardmax #(
    .dataWidth(dataWidth), 
    .numOutputs(10)
) hardmax_inst (
    // Inputs
    .clk(clk),
    .reset(reset),
    .dataIn(NNout),
    .enable(NNoutValid),
    // Outputs
    .maxIndex(maxIndex),
    .maxValue(maxValue),
    .maxValid(maxValid)
);
    
endmodule