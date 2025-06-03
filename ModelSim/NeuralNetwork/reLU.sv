module reLU #(
    parameter   sumWidth = 32, sumIntWidth = 15, sumFracWidth = 17,
                dataWidth = 16, dataIntWidth = 6, dataFracWidth = 10,
                weightWidth = 8, weightIntWidth = 1, weightFracWidth = 7
)(
    input   logic   [sumWidth-1:0]     dataIn,
    output  logic   [dataWidth-1:0]     dataOut
);

    always_comb begin
        if (dataIn[sumWidth-1] == 0) begin // positive input
            if (dataIn >= {1'b0, {(dataIntWidth + sumFracWidth - 1){1'b1}}}) begin
                dataOut = {1'b0, {(dataWidth-1){1'b1}}}; // Saturate to Max
            end
            else begin
                dataOut = dataIn[(dataWidth + weightFracWidth - 1) : weightFracWidth];
            end
        end 
        else begin
            dataOut = 0;
        end
    end
endmodule: reLU