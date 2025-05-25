`timescale 1ns / 1ps

module testbench ( );

	parameter CLOCK_PERIOD = 10;

	reg CLOCK_50;
    reg [0:0] KEY;
    wire [6:0] HEX0;

	initial begin
        CLOCK_50 <= 1'b0;
	end // initial
	always @ (*)
	begin : Clock_Generator
		#((CLOCK_PERIOD) / 2) CLOCK_50 <= ~CLOCK_50;
	end
	
	initial begin
        KEY[0] <= 1'b0;
        #10 KEY[0] <= 1'b1;
	end // initial
	
	//module U1 (KEY, CLOCK_50, HEX0);

endmodule
