module neuron #(
    parameter layerNumber = 0, neuronNumber = 0, numWeights = 256, dataWidth = 8, weightIntWidth = 1, biasFile = "b_l0_n0.mif", weightFile = "w_l0_n0.mif"
)   (
    input   logic                       clk,
    input   logic                       rst,
    input   logic                       neuronIn,
    input   logic                       neuronValid,
    input   logic                       weightValid,
    input   logic    [31:0]             weightData,
    input   logic    [31:0]             biasData,
    input   logic    [31:0]             config_layer_number,
    input   logic    [31:0]             config_neuron_number,

    output  logic    [dataWidth-1:0]    neuronOut,
    output  logic                       neuronOutValid,
    );

    // Parameters
    parameter addressWidth = $clog2(numWeights);

    // Weights Module
    weights #(
        .numWeights(numWeights),
        .neuronNumber(neuronNumber),
        .layerNumber(layerNumber),
        .addressWidth(addressWidth),
        .dataWidth(dataWidth),
        .weightFile(weightFile)
    ) Weight (
        .clk(clk),
        .readEn(readEn),
        .writeEn(writeEn),
        .addr(addr),
        .dataIn(dataIn),
        .dataOut(dataOut)
    );
    
endmodule: neuron