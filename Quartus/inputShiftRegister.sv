module inputShiftRegister #(
    parameter numInputs = 784, dataWidth = 16, dataFracWidth = 8, dataIntWidth = 8
) ( 
    input  logic                                reset,
    input  logic                                serialClock,
    input  logic                                serialData,

    output  logic [numInputs * dataWidth - 1:0]  dataOut
);
    logic [numInputs - 1:0]     internalRegister;
    

    always_ff @(posedge serialClock or posedge reset) begin
        if (reset) begin
            internalRegister <= 0;
        end
        else begin
            internalRegister <= {internalRegister[numInputs - 2 : 0], serialData};
        end
    end

    integer i;
    always_comb begin   // Make output signed fixed point Qm.n format
        for (i = 0; i < numInputs; i = i + 1) begin
            if (internalRegister[i])
                dataOut[dataWidth*i +: dataWidth] = 16'h2000; // 32.0
            else
                dataOut[dataWidth*i +: dataWidth] = '0; // 0.0
        end
    end
endmodule