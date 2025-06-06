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
    logic [31:0]        dataOut;

	initial begin
        CLOCK_50 <= 1'b0;
        serialClock <= 1'b0;
	end // initial
	always @ (*)
	begin : Clock_Generator
		#((CLOCK_PERIOD) / 2) CLOCK_50 <= ~CLOCK_50;
	end
    always @ (*)
	begin : Clock_Generator_2
		#((serialClockPeriod) / 2) serialClock <= ~serialClock;
	end
    
	
	initial begin
        pushBuffer <= 1'b0;
        reset <= 1'b0;
        #10
        reset <= 1'b1;
		#20 
		reset <= 1'b0;
		serialData <= 1'b0;
		#15
		serialData <= 1'b1;
		#15
        serialData <= 1'b1;
		#15
        serialData <= 1'b0;
		#15
        serialData <= 1'b0;
		#15
        serialData <= 1'b1;
		#15
        serialData <= 1'b1;
		#15
        serialData <= 1'b1;
		#15
        serialData <= 1'b0;
		#15
        serialData <= 1'b1;
		#15
        serialData <= 1'b0;
		#15
        serialData <= 1'b1;
		#15
        serialData <= 1'b0;
		#15
        serialData <= 1'b0;
		#15
        serialData <= 1'b1;
		#15
        serialData <= 1'b0;
		#15
        serialData <= 1'b1;
		#15
        serialData <= 1'b1;
		#15
        pushBuffer <= 1'b1;
        #10
        pushBuffer <= 1'b0;
	end // initial

    inputShiftRegister # (
        .numInputs(8),
        .dataWidth(4)
    ) U1 (
        .reset(reset),
        .CLOCK_50(CLOCK_50),
        .serialClock(serialClock),
        .serialData(serialData),
        .pushBuffer(pushBuffer),

        .dataOut(dataOut)
    );
endmodule
