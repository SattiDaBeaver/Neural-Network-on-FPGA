module inputSerializer #(
    parameter numInputs = 16, dataWidth = 16, counterWidth = $clog2(numInputs)
) (
    input    logic                              clk,
    input    logic                              reset,
    input    logic                              enable,
    input    logic [dataWidth*numInputs-1:0]    serializerIn, // Serialized input data

    output   logic [counterWidth-1:0]           counterOut,
    output   logic                              counterValid,
    output   logic [dataWidth-1:0]              serializerOut // Output data 
    );

    logic delay;

    // Delay to match Neuron FSM
    always_ff @ (posedge clk) begin
        if (reset) begin
            delay <= 0;
        end 
        else if (enable && !delay) begin
            delay <= delay + 1;
        end
    end

    // Counter Logic
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

    // Output Data Logic
    always_comb begin
        if (!counterValid && enable && delay) begin
            serializerOut = serializerIn[(counterOut + 1) * dataWidth - 1 -: dataWidth]; // Extract the next input data
        end else begin
            serializerOut = '0; // Default value when not valid
        end
    end
    
endmodule