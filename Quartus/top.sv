`define PRETRAINED

module top (
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
    logic   [5:0]   dataIn;
    logic   [5:0]   dataOut;
    logic   [7:0]   addr;
    logic           readEn;
    logic           writeEn;


	parameter CLOCK_PERIOD = 10;

	logic CLOCK_50;

	// Layer Test
	logic 				reset;
	logic [784*16-1:0] 	NNin;
	logic [10*16-1:0] 	NNout;
	logic 			    NNvalid;
	logic 			    NNoutValid;
    logic [3:0]         maxIndex;
    logic [15:0]        maxValue;
    logic               maxValid;

    NeuralNetwork #(
        .numInputs(784), 
        .numOutputs(10), 
        .L0neurons(16), 
        .L1neurons(10),
        .dataWidth(16), 
        .dataIntWidth(8),
        .dataFracWidth(8),
        .weightWidth(16),
        .weightIntWidth(8),
        .weightFracWidth(8)
    ) nn (
        .clk(CLOCK_50),
        .reset(reset),
        .NNin(NNin),
        .NNvalid(NNvalid),
        .NNout(NNout),
        .NNoutValid(NNoutValid),
        .maxIndex(maxIndex),
        .maxValid(maxValid),
        .maxValue(maxValue)
    );

endmodule
