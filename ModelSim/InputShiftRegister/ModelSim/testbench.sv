`timescale 1ns / 1ps
// `define PRETRAINED

module testbench ( );

	parameter CLOCK_PERIOD = 10;

	logic CLOCK_50;
    logic [0:0] KEY;
    logic [6:0] HEX0;

	// Neuron Test
	logic 				reset;
	logic [7:0] 		neuronIn;
	logic [7:0] 		neuronOut;
	logic 				neuronValid;
	logic 				neuronOutValid;

    // Input Counter
    logic [3:0]       inputCounterOut;
    logic             inputCounterValid;

    // Weight Memory
    logic [7:0]       weightMem [0:255];

	initial begin
        CLOCK_50 <= 1'b0;
	end // initial
	always @ (*)
	begin : Clock_Generator
		#((CLOCK_PERIOD) / 2) CLOCK_50 <= ~CLOCK_50;
	end
	
	initial begin
        reset <= 1'b0;
        #10
        reset <= 1'b1;
		#10 
		reset <= 1'b0;
		neuronIn <= 8'h02;
		#10
		neuronValid <= 1'b1;
	end // initial

    inputCounter #(
        .numInputs(16), 
        .counterWidth()
    ) U1 (
        .clk(CLOCK_50),
        .reset(reset),
        .enable(neuronValid),
        .counterOut(inputCounterOut),
        .counterValid(inputCounterValid)
    );
	
	// neuron #(
    //     .layerNumber(0),
    //     .neuronNumber(0),
    //     .numWeights(16),
    //     .dataWidth(8),
    //     .weightIntWidth(4),
    //     .biasFile("b_l0_n0.mif"),
    //     .weightFile("w_l0_n0.mif")
    // ) U1 (
    //     .clk(CLOCK_50),
    //     .reset(reset),
    //     .neuronIn(neuronIn),
    //     .neuronValid(neuronValid),
    //     .weightValid(),
    //     .weightWriteEn(),
    //     .biasWriteEn(),
    //     .weightData(32'h0),
    //     .biasData(32'h0),
    //     .config_layer_number(32'h0),
    //     .config_neuron_number(32'h0),
    //     .neuronOut(neuronOut),
    //     .neuronOutValid(neuronOutValid)
    // );

endmodule
