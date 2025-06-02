module reLU #(
    parameter sumWidth = 24, dataWidth = 8
)(
    input   logic   [sumWidth-1:0]     dataIn,
    output  logic   [dataWidth-1:0]     dataOut
);

    always_comb begin
        if (dataIn[sumWidth-1] == 0) begin
            if (dataIn[sumWidth-1:2*dataWidth-1] != 0) begin
                dataOut = 8'h7F;
            end
            else begin
                dataOut = dataIn[2*dataWidth-2-:dataWidth];
            end
        end 
        else begin
            dataOut = 0;
        end
    end
endmodule: reLU