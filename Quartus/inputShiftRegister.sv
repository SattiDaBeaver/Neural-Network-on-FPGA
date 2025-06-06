module inputShiftRegister #(
    parameter numInputs = 784, dataWidth = 16
) ( 
    input  logic                                reset,
    input  logic                                CLOCK_50,
    input  logic                                serialClock,
    input  logic                                serialData,
    input  logic                                pushBuffer,

    output  logic [numInputs * dataWidth - 1:0]  dataOut
);
    logic [numInputs * dataWidth - 1:0]     internalRegister;
    

    always_ff @(posedge serialClock or posedge reset) begin
        if (reset) begin
            internalRegister <= 0;
        end
        else begin
            internalRegister <= {internalRegister[numInputs * dataWidth - 2 : 0], serialData};
        end
    end

    always_ff @(posedge CLOCK_50) begin
        if (reset) begin
            dataOut <= 0;
        end
        else begin
            if (pushBuffer == 1'b1) begin
                dataOut <= internalRegister;
            end
        end 
    end
    
endmodule