module inputShiftRegister #(
    parameter numInputs = 784, dataWidth = 16
) (
    input  logic                                CLOCK_50;
    input  logic                                serialClock;
    input  logic                                serialData;
    input  logic                                pushBuffer;

    output  logic [numInputs * dataWidth - 1:0]  dataOut;
);
    logic [numInputs * dataWidth - 1:0]     internalRegister;
    
    
endmodule