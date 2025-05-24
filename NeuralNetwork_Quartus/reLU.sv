module reLU #(
    parameter sumWidth = 16, dataWidth = 8
)(
    input   logic   [sumWidth-1:0]     dataIn,
    output  logic   [sumWidth-1:0]     dataOut
);

    always_comb begin
        if (dataIn[sumWidth-1] == 0) begin
            dataOut = dataIn[sumWidth-1-:dataWidth];
        end 
        else begin
            dataOut = 0;
        end
    end
endmodule: reLU