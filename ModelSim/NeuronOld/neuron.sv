module neuron #(
    parameter layerNumber = 0, neuronNumber = 0, numWeights = 256, dataWidth = 8, weightIntWidth = 4, biasFile = "b_l0_n0.mif", weightFile = "w_l0_n0.mif"
)   (
    input   logic                       clk,
    input   logic                       reset,
    input   logic    [dataWidth-1:0]    neuronIn,
    input   logic                       neuronValid,
    input   logic                       weightValid,
    input   logic                       weightWriteEn,
    input   logic                       biasWriteEn,
    input   logic    [31:0]             weightData,
    input   logic    [31:0]             biasData,
    input   logic    [31:0]             config_layer_number,
    input   logic    [31:0]             config_neuron_number,

    output  logic    [dataWidth-1:0]    neuronOut,
    output  logic                       neuronOutValid
    );

    // Parameters
    parameter addressWidth = $clog2(numWeights);
    parameter safetyBits = 3;

    // Internal Nets
    logic    [dataWidth-1:0]                    weightOut;
    logic    [safetyBits*dataWidth-1:0]         multOut;    // Multiplier output: 2 * dataWidth
    logic    [safetyBits*dataWidth-1:0]         adderOut;
    logic    [safetyBits*dataWidth-1:0]         adderOutWire;
    logic    [dataWidth-1:0]                    biasOut [0:0];
    logic    [safetyBits*dataWidth-1:0]         sumOut;
    logic    [safetyBits*dataWidth-1:0]         sumOutWire;
    logic    [dataWidth-1:0]                    reluOut;

    logic    [addressWidth-1:0]                 weightAddress;
    logic                                       readEn;

    // Control Signals
    logic                                       MACenable;
    logic                                       activationEnable;

    typedef enum logic [1:0] {
        IDLE,
        MAC_STATE,
        DELAY_STATE,
        OUTPUT_STATE
    } state_t;

    state_t state;
    state_t nextState;

    // Control Logic FSM
    always_comb begin
        nextState         = state;
        MACenable         = 0;
        readEn            = 0;
        activationEnable  = 0;
        case (state)
            IDLE: begin
                if (neuronValid) begin
                    nextState = MAC_STATE;
                end
                else begin
                    nextState = IDLE;
                end
            end

            MAC_STATE: begin
                MACenable =  1;
                readEn = 1;
                if (weightAddress == (numWeights - 1)) begin
                    nextState = DELAY_STATE;
                end
                else begin
                    // readEn = 1;
                end
            end

            DELAY_STATE: begin
                readEn = 0;
                nextState = OUTPUT_STATE;
            end

            OUTPUT_STATE: begin
                activationEnable = 1;
                nextState = OUTPUT_STATE;
            end

            default: begin
                nextState = IDLE;
            end
        endcase
    end

    always_ff @ (posedge clk) begin
        if (reset) begin
            weightAddress <= 0;
            state <= IDLE;
        end 
        else begin
            state <= nextState;
        end
        if (state == MAC_STATE) begin
            if (weightAddress == (numWeights - 1)) begin
                    weightAddress <= 0;
            end
            else begin
                    weightAddress <= weightAddress + 1;
            end
        end
        
    end


    // Multiply and Accumulate (MAC): Dot Product
    // Multiplier
    always_comb begin
        multOut = $signed(weightOut) * $signed(neuronIn);
    end

    // Adder
    assign adderOutWire = $signed(multOut) + $signed(adderOut);
    always_ff @ (posedge clk) begin
        if (reset) begin
            adderOut <= 0;
        end
        else if (MACenable) begin
            if ((multOut[safetyBits*dataWidth-1] == 1) && (adderOut[safetyBits*dataWidth-1] == 1) && (adderOutWire[safetyBits*dataWidth-1] == 0)) begin
                adderOut <= {1'b1,{(safetyBits*dataWidth-1){1'b0}}}; // Saturate to min value
            end
            else if ((multOut[safetyBits*dataWidth-1] == 0) && (adderOut[safetyBits*dataWidth-1] == 0) && (adderOutWire[safetyBits*dataWidth-1] == 1)) begin
                adderOut <= {1'b0,{(safetyBits*dataWidth-1){1'b1}}}; // Saturate to max value
            end
            else begin
                adderOut <= $signed(multOut) + $signed(adderOut);
            end
        end
        else begin
            adderOut <= adderOut;
        end
    end

    // Bias Adder
    initial begin
        $readmemb(biasFile, biasOut);
    end

    always_ff @ (posedge clk) begin
        if (biasWriteEn) begin
            biasOut[0] <= biasData;
        end
        else begin
            biasOut[0] <= biasOut[0];
        end
    end


    assign sumOutWire = $signed(adderOut) + ($signed(biasOut[0]) <<< 7);
    always_comb begin
        if ((biasOut[0][dataWidth-1] == 1) && (adderOut[safetyBits*dataWidth-1] == 1) && (sumOutWire[safetyBits*dataWidth-1] == 0)) begin
            sumOut = {1'b1,{(safetyBits*dataWidth-1){1'b0}}}; // Saturate to min value
        end
        else if ((biasOut[0][dataWidth-1] == 0) && (adderOut[safetyBits*dataWidth-1] == 0) && (sumOutWire[safetyBits*dataWidth-1] == 1)) begin
            sumOut = {1'b0,{(safetyBits*dataWidth-1){1'b1}}}; // Saturate to max value
        end
        else begin
            sumOut = sumOutWire;
        end
    end
    
    // Acitivation Function
    reLU #(
        .sumWidth(2*dataWidth),
        .dataWidth(dataWidth)
    ) ReLU (
        .dataIn(sumOut >>> 7),
        .dataOut(reluOut)
    );

    // Output
    always_ff @ (posedge clk) begin
        if (reset) begin
            neuronOut <= 0;
            neuronOutValid <= 0;
        end
        else if (activationEnable) begin
            neuronOut <= reluOut;
            neuronOutValid <= 1;
        end
        else begin
            neuronOut <= neuronOut;
            neuronOutValid <= 0;
        end
    end

    // Weights Module
    weights #(
        .numWeights(numWeights),
        .neuronNumber(neuronNumber),
        .layerNumber(layerNumber),
        .addressWidth(addressWidth),
        .dataWidth(dataWidth),
        .weightFile(weightFile)
    ) Weight (
        .clk(clk),
        .readEn(readEn),
        .writeEn(writeEn),
        .addr(weightAddress),
        .dataIn(weightData),
        .dataOut(weightOut)
    );
    
endmodule: neuron