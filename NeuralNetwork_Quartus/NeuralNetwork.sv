`define PRETRAINED

module NeuralNetwork(
	input  logic    [9:0]   SW,
	input  logic    [1:0]   KEY,
	input  logic            CLOCK_50,

	output logic    [6:0]   HEX5,
	output logic    [6:0]   HEX4,
	output logic    [6:0]   HEX3,
	output logic    [6:0]   HEX2,
	output logic    [6:0]   HEX1,
	output logic    [6:0]   HEX0,
	output logic    [9:0]   LEDR 
);
    // Internal signals
    logic   [9:0]   dataIn;
    logic   [5:0]   dataOut;
    logic   [7:0]   addr;
    logic           readEn;
    logic           writeEn;

    layer layer1 ();

    // Neuron Test
    neuron #(
        .layerNumber(0),
        .neuronNumber(0),
        .numWeights(16),
        .dataWidth(8),
        .weightIntWidth(4),
        .biasFile("bias_L0_N0.mif"),
        .weightFile("weight_L0_N0.mif")
    ) Neuron (
        .clk(CLOCK_50),
        .reset(~KEY[0]),
        .neuronIn(SW[7:0]),
        .neuronValid(~KEY[1]),
        .weightValid(),
        .weightWriteEn(),
        .biasWriteEn(),
        .weightData(32'h0),
        .biasData(32'h0),
        .config_layer_number(32'h0),
        .config_neuron_number(32'h0),
        .neuronOut(LEDR[7:0]),
        .neuronOutValid(LEDR[9])
    );

    // // Multiplier Test
    // always_comb begin
    //     LEDR[9:0] = $signed(SW[4:0]) * $signed(SW[9:5]);
    // end

    // assign readEn = ~KEY[0];
    // assign writeEn = ~KEY[1];
    // // assign dataIn = SW[9:0];
    // assign addr = SW[7:0];
    // assign LEDR[7:0] = dataOut;

    // // Instantiate the weights module

    // weights #(
    //     .numWeights(256),
    //     .neuronNumber(0),
    //     .layerNumber(1),
    //     .addressWidth(8),
    //     .dataWidth(8),
    //     .weightFile("w_l0_n0.mif")
    // ) Weight (
    //     .clk(CLOCK_50),
    //     .readEn(readEn),
    //     .writeEn(writeEn),
    //     .addr(addr),
    //     .dataIn(dataIn),
    //     .dataOut(dataOut)
    // );

    // reLU #(
    //     .sumWidth(10),
    //     .dataWidth(6)
    // ) ReLU (
    //     .dataIn(dataIn),
    //     .dataOut(dataOut)
    // );

endmodule: NeuralNetwork