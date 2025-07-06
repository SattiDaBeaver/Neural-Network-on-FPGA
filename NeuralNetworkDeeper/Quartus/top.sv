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
	output logic    [9:0]   LEDR, 

    input  logic    [15:0]  ARDUINO_IO    
);

    // Neural Network Parameters
    parameter   dataWidth = 16, dataIntWidth = 8, dataFracWidth = 8;
    parameter   weightWidth = 16, weightIntWidth = 8, weightFracWidth = 8;
    parameter   numInputs = 784, numOutputs = 10;
    parameter   L0neurons = 16, L1neurons = 16, L2neurons = 16, L3neurons = 10;

    // Input Parameters
    parameter   inputMemSize = 2;
    parameter   addressWidth = $clog2(inputMemSize);

    // Internal signals


	// Layer Test
	logic 				reset;
    logic               NNreset;
	logic [784*16-1:0] 	NNin;

	logic [10*16-1:0] 	NNout;
	logic 			    NNvalid;
	logic 			    NNoutValid;
    logic [3:0]         maxIndex;
    logic [3:0]         maxIndexReg;
    logic [15:0]        maxValue;
    logic               maxValid;

    // Input Shift Register
    logic               serialClock;
    logic               serialData;
    logic               pushBuffer;

    assign serialClock = ARDUINO_IO[0];
    assign serialData = ARDUINO_IO[1];

    assign reset = ~KEY[1];
    assign addr  = SW[9:0];

    // Max Index Reg
    always_ff @(posedge CLOCK_50) begin
        if (reset) begin
            maxIndexReg <= 0;
        end
        else if (maxValid) begin
            maxIndexReg <= maxIndex;
        end
    end

    // Control FSM
    typedef enum logic [2:0] {
        IDLE, RESET_NN, DELAY_1, START_NN, WAIT_NN, DELAY_2
    } state_t;

    state_t state, nextState;

    always_ff @(posedge CLOCK_50) begin
        if (reset) begin
            state <= IDLE;
        end
        else begin
            state <= nextState;
        end
    end

    always_comb begin
        nextState = state;  // Default to hold state
        NNreset = 1'b0;
        NNvalid = 1'b0;
        case (state)
            IDLE: begin
                nextState = RESET_NN;
            end
            RESET_NN: begin
                NNreset = 1'b1;
                nextState = DELAY_1;
            end
            DELAY_1: begin
                    nextState = START_NN;
            end
            START_NN: begin
                NNvalid = 1'b1;
                nextState = WAIT_NN;
            end
            WAIT_NN: begin
                if (maxValid) begin
                    NNvalid = 1'b0;
                    nextState = DELAY_2;
                end
                else begin
                    NNvalid = 1'b1;
                    nextState = WAIT_NN;
                end
            end
            DELAY_2: begin
                nextState = IDLE;
            end
            default: nextState = IDLE;
        endcase
    end

    // Module Instantiation

    inputShiftRegister # (
        .numInputs(numInputs),
        .dataWidth(dataWidth),
        .dataFracWidth(dataFracWidth),
        .dataIntWidth(dataIntWidth)
        ) shiftReg (
        .reset(reset),
        .serialClock(serialClock),
        .serialData(serialData),

        .dataOut(NNin)
    );

    NeuralNetwork #(
        .numInputs(numInputs), 
        .numOutputs(numOutputs), 
        .L0neurons(L0neurons), 
        .L1neurons(L1neurons),
        .L2neurons(L2neurons),
        .L3neurons(L3neurons),
        .dataWidth(dataWidth), 
        .dataIntWidth(dataIntWidth),
        .dataFracWidth(dataFracWidth),
        .weightWidth(weightWidth),
        .weightIntWidth(weightIntWidth),
        .weightFracWidth(weightFracWidth)
        ) nn (
        .clk(CLOCK_50),
        .reset(NNreset),
        .NNin(NNin),
        .NNvalid(NNvalid),

        .NNout(NNout),
        .NNoutValid(NNoutValid),
        .maxIndex(maxIndex),
        .maxValid(maxValid),
        .maxValue(maxValue)
    );

    assign HEX1 = 7'h7F;
    assign HEX2 = 7'h7F;
    assign HEX3 = 7'h7F;
    assign HEX4 = 7'h7F;
    assign HEX5 = 7'h7F;

    hex7seg hex0 (
        .hex(maxIndexReg),
        .display(HEX0)
    );

endmodule: top


module hex7seg (hex, display);
    input [3:0] hex;
    output [6:0] display;

    reg [6:0] display;

    /*
     *       0  
     *      ---  
     *     |   |
     *    5|   |1
     *     | 6 |
     *      ---  
     *     |   |
     *    4|   |2
     *     |   |
     *      ---  
     *       3  
     */
    always @ (hex)
        case (hex)
            4'h0: display = 7'b1000000;
            4'h1: display = 7'b1111001;
            4'h2: display = 7'b0100100;
            4'h3: display = 7'b0110000;
            4'h4: display = 7'b0011001;
            4'h5: display = 7'b0010010;
            4'h6: display = 7'b0000010;
            4'h7: display = 7'b1111000;
            4'h8: display = 7'b0000000;
            4'h9: display = 7'b0011000;
            4'hA: display = 7'b0001000;
            4'hB: display = 7'b0000011;
            4'hC: display = 7'b1000110;
            4'hD: display = 7'b0100001;
            4'hE: display = 7'b0000110;
            4'hF: display = 7'b0001110;
        endcase
endmodule: hex7seg
