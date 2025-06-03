module hardmax #(
    parameter dataWidth = 16, numOutputs = 10, addressWidth = $clog2(numOutputs)
) (
    input   logic                               clk,
    input   logic                               reset,
    input   logic   [dataWidth*numOutputs-1:0]  dataIn,
    input   logic                               enable,

    output  logic   [addressWidth-1:0]          maxIndex,
    output  logic   [dataWidth-1:0]             maxValue,
    output  logic                               maxValid
);

    // Internal Nets
    logic [addressWidth-1:0]    indexCounter;
    logic [dataWidth-1:0]       currentValue;

    //assign currentValue = dataIn[(numOutputs - 1 - indexCounter)*dataWidth +: dataWidth];
    assign currentValue = dataIn[indexCounter*dataWidth +: dataWidth];

    always_ff @ (posedge clk) begin
        if (reset) begin
            maxValue <= '0;
            indexCounter <= '0;
            maxValid <= 0;
        end 
        else if (enable && !maxValid) begin
            if (currentValue > maxValue || indexCounter == 0) begin
                maxValue <= currentValue;
                maxIndex <= indexCounter;
            end

            if (indexCounter == numOutputs - 1) begin
                indexCounter <= '0; // Reset counter after last output
                maxValid <= 1;
            end 
            else begin
                indexCounter <= indexCounter + 1; // Increment counter
            end
        end 
    end
endmodule: hardmax