// `define PRETRAINED

module NeuralNetwork(
	input logic     [9:0]   SW,
	input logic     [1:0]   KEY,
	input logic             CLOCK_50,

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

    assign readEn = ~KEY[0];
    assign writeEn = ~KEY[1];
    assign dataIn = SW[9:0];
    // assign addr = SW[7:0];
    assign LEDR[5:0] = dataOut;

    // Instantiate the weights module

    // weights #(
    //     .numWeights(256),
    //     .neuronNumber(0),
    //     .layerNumber(1),
    //     .addressWidth(8),
    //     .dataWidth(6),
    //     .weightFile("w_n0_l1.mif")
    // ) Weight (
    //     .clk(CLOCK_50),
    //     .readEn(readEn),
    //     .writeEn(writeEn),
    //     .addr(addr),
    //     .dataIn(dataIn),
    //     .dataOut(dataOut)
    // );

    reLU #(
        .sumWidth(10),
        .dataWidth(6)
    ) ReLU (
        .dataIn(dataIn),
        .dataOut(dataOut)
    );

endmodule: NeuralNetwork