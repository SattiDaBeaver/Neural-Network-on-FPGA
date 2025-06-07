`timescale 1ns / 1ps
// `define PRETRAINED

module testbench ( );

	parameter CLOCK_PERIOD = 10;
    parameter serialClockPeriod = 15;

	logic CLOCK_50;
    logic serialClock;

	// Shift Register Test
	logic 				reset;
    logic               pushBuffer;
	logic               serialData;
    logic [32*4-1:0]        dataOut;
    logic [31:0]       data;

	initial begin
        CLOCK_50 <= 1'b0;
        serialClock <= 1'b0;
	end // initial
	always @ (*)
	begin : Clock_Generator
		#((CLOCK_PERIOD) / 2) CLOCK_50 <= ~CLOCK_50;
	end
    // always @ (*)
	// begin : Clock_Generator_2
	// 	#((serialClockPeriod) / 2) serialClock <= ~serialClock;
	// end
    
	integer i;
	initial begin
        //data <= 'b0000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000011100000000000000000000000011111111111111100000000000000000111111111111000000000000000000000000011000000000000000000000000001100000000000000000000000001100000000000000000000000001110000000000000000000000000110000000000000000000000000111000000000000000000000000011000000000000000000000000001100000000000000000000000001100000000000000000000000001110000000000000000000000001110000000000000000000000000110000000000000000000000000110000000000000000000000000111000000000000000000000000011100000000000000000000000001110000000000000000000000000110000000000000000000000000000000000000000000;
        data <= 32'hFF203040;
        pushBuffer <= 1'b0;
        reset <= 1'b0;
        serialClock <= 1'b0;
        #10
        reset <= 1'b1;
		#20 
		reset <= 1'b0;
        #20
		// shift in data (LSB first or MSB first depending on DUT)
        for (i = 0; i < 32; i = i + 1) begin
            serialData = data[i];  // sends LSB first (data[0] → first)
            
            #5 serialClock = 1;  // rising edge
            #5 serialClock = 0;  // falling edge
        end
	end // initial

    inputShiftRegister # (
        .numInputs(32),
        .dataWidth(4),
        .dataFracWidth(2),
        .dataIntWidth(2)
    ) U1 (
        .reset(reset),
        .serialClock(serialClock),
        .serialData(serialData),

        .dataOut(dataOut)
    );
endmodule
