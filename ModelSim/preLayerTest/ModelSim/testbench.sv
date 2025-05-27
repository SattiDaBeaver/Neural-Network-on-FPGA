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

    // Input Counter
    logic [3:0]       inputCounterOut;
    logic             inputCounterValid;
    logic [127:0]     serializerIn; // Serialized input data
    logic [7:0]       serializerOut; // Output data

    initial begin
        $readmemb("w_l0_n0.mif", weightMem);
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
        serializerIn <= 128'h0F0E0D0C0B0A09080706050403020100; // Example input data
        #10
        reset <= 1'b1;
		#10 
		reset <= 1'b0;
		neuronIn <= 8'h02;
		#10
		neuronValid <= 1'b1;
	end // initial

    inputSerializer #(
        .numInputs(16), 
        .counterWidth()
    ) U2 (
        .clk(CLOCK_50),
        .reset(reset),
        .enable(neuronValid),
        .counterOut(inputCounterOut),
        .counterValid(inputCounterValid),
        .serializerIn(serializerIn),
        .serializerOut(serializerOut)
    );
	
	neuron #(
        .layerNumber(0),
        .neuronNumber(0),
        .numWeights(16),
        .dataWidth(8),
        .weightIntWidth(4),
        .biasFile("b_l0_n0.mif"),
        .weightFile("w_l0_n0.mif")
    ) U1 (
        .clk(CLOCK_50),
        .reset(reset),
        .neuronIn(serializerOut),
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
