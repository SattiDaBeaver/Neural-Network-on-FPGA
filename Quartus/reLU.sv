module reLU #(
    parameter sumWidth = 16, dataWidth = 8
)(
    input   logic   [sumWidth-1:0]     dataIn,
    output  logic   [dataWidth-1:0]     dataOut
);

    always_comb begin
        if (dataIn[sumWidth-1] == 0) begin
            if (dataOut >= 8'h7F) begin
                dataOut = 8'h7F;
            end
            else begin
                dataOut = dataIn[dataWidth-1:0];
            end
        end 
        else begin
            dataOut = 0;
        end
    end
endmodule: reLU