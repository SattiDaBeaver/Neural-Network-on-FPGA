module inputCounter #(
    parameter numInputs = 16, counterWidth = $clog2(numInputs+1)
) (
    input    logic                      clk,
    input    logic                      reset,
    input    logic                      enable,
    output   logic [counterWidth-1:0]   counterOut,
    output   logic                      counterValid
    );

    logic delay;

    always_ff @ (posedge clk) begin
        if (reset) begin
            delay <= 0;
        end 
        else if (enable && !delay) begin
            delay <= delay + 1;
        end
    end

    always_ff @ (posedge clk) begin
        if (reset) begin
            counterOut <= 0;
            counterValid <= 0;
        end 
        else begin
            if (counterOut == numInputs - 1)begin
                counterOut <= 0; // Reset to zero after reaching the maximum count
                counterValid <= 1; // Set valid signal when counter is reset
            end
            else if (!counterValid && enable && delay) begin
                counterOut <= counterOut + 1;
            end 
        end
    end
    
endmodule