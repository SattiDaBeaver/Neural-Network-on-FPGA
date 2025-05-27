`timescale 1ns / 1ps
// `define PRETRAINED

module testbench ( );

	parameter CLOCK_PERIOD = 10;

	logic CLOCK_50;

	// Layer Test
	logic 				reset;
	logic [784*8-1:0] 		layerIn;
	logic [16*8-1:0] 		layerOut;
	logic 				layerValid;
	logic 				layerOutValid;

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
        layerIn <= 'h0F0E0D0C0B0A09080706050403020100; // Example input data
        #10
        reset <= 1'b1;
		#10 
		reset <= 1'b0;
		#10
		layerValid <= 1'b1;
	end // initial

    layer #(
        .layerNumber(0),
        .dataWidth(0),
        .numInputs(784),
        .numNeurons(16)
    ) U1 (
        .clk(CLOCK_50),
        .reset(reset),
        .layerIn(layerIn),
        .layerValid(layer),
        .layerOut(layerOut),
        .layerOutValid(layerOutValid)
    );

endmodule
