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
        $readmemb("weights/weight_L0_N0.mif", weightMem);
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
        layerIn <= {392{16'h0203}};
        //'h297F3C12A49D562B00FF1A4E9063C1887634204599AFBE1867D32C4AE207851E543DA8910CF2B46B1D7E39C7E14028ABFC0F3E5D2A8473656E89A210B5374F9B0B112233445566778899AABBCCDDEEFFFEDCBA98765432100123456789ABCDEFC0C1C2C3C4C5C6C7C8C9CACBCCCDCECFD0D1D2D3D4D5D6D7D8D9DADBDCDDDEDFE0E1E2E3E4E5E6E7E8E9EAEBECEDEEF0F1F2F3F4F5F6F7F8F9FAFBFCFDFEFF000102030405060708090A0B0C0D0E0F101112131415161718191A1B1C1D1E1F202122232425262728292A2B2C2D2E2F303132333435363738393A3B3C3D3E3F404142434445464748494A4B4C4D4E4F505152535455565758595A5B5C5D5E5F606162636465666768696A6B6C6D6E6F707172737475767778797A7B7C7D7E7F808182838485868788898A8B8C8D8E8F909192939495969798999A9B9C9D9E9FA0A1A2A3A4A5A6A7A8A9AAABACADAEAFB;
        #10
        reset <= 1'b1;
		#10 
		reset <= 1'b0;
		#10
		layerValid <= 1'b1;
	end // initial

    layer #(
        .layerNumber(0),
        .dataWidth(8),
        .numInputs(784),
        .numNeurons(16)
    ) U1 (
        .clk(CLOCK_50),
        .reset(reset),
        .layerIn(layerIn),
        .layerValid(layerValid),
        .layerOut(layerOut),
        .layerOutValid(layerOutValid)
    );

endmodule
