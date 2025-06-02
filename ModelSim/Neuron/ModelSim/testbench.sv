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

    // Weight Memory
    logic [7:0]       weightMem [0:255];

    initial begin
        $readmemb("weight_L0_N0.mif", weightMem);
        $display("Loaded weights:");
        for (int i = 0; i < 16; i++)
            $display("weightMem[%0d] = %b", i, weightMem[i]);
    end

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
		neuronIn <= 8'h70; // Test with a positive input
		#10
		neuronValid <= 1'b1;
        #300

        reset <= 1'b0;
        #10
        reset <= 1'b1;
		#10 
        reset <= 1'b0;
		neuronIn <= 8'h8f; // Test with a negative input
		#10
		neuronValid <= 1'b1;
        
	end // initial
	
	neuron #(
        .layerNumber(0),
        .neuronNumber(0),
        .numWeights(16),
        .dataWidth(8),
        .weightIntWidth(4),
        .biasFile("bias_L0_N0.mif"),
        .weightFile("weight_L0_N0.mif")
    ) U1 (
        .clk(CLOCK_50),
        .reset(reset),
        .neuronIn(neuronIn),
        .neuronValid(neuronValid),
        .weightValid(),
        .weightWriteEn(),
        .biasWriteEn(),
        .weightData(32'h0),
        .biasData(32'h0),
        .config_layer_number(32'h0),
        .config_neuron_number(32'h0),
        .neuronOut(neuronOut),
        .neuronOutValid(neuronOutValid)
    );

endmodule
