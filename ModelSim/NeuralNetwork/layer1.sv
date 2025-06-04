module layer1 #(
    parameter   layerNumber = 1, numInputs = 16, numNeurons = 10,
                dataWidth = 16, dataIntWidth = 6, dataFracWidth = 10, 
                weightWidth = 16, weightIntWidth = 6, weightFracWidth = 10
) (
    input   logic                               clk,
    input   logic                               reset, 
    input   logic   [dataWidth*numInputs-1:0]   layerIn,
    input   logic                               layerValid,

    output  logic   [dataWidth*numNeurons-1:0]  layerOut,
    output  logic                               layerOutValid
);

    // Parameters
    parameter counterWidth = $clog2(numInputs+1);

    // Internal Nets
    logic [dataWidth-1:0] serializerOut;
    logic counterValid;

    logic [numNeurons-1:0] neuronOutValid;

    // Serializer Instance
    inputSerializer #(
        .numInputs(numInputs), 
        .dataWidth(dataWidth), 
        .counterWidth(counterWidth)
    ) serializer (
        .clk(clk),
        .reset(reset),
        .enable(layerValid),
        .serializerIn(layerIn),
        .counterValid(counterValid),
        .serializerOut(serializerOut)
    );

    // Neuron Instances
    genvar i;
    generate
        for (i = 0; i < numNeurons; i = i + 1) begin : gen_neurons
            localparam string weightFile = 
                (i == 0)  ? "weights/weight_L1_N0.mif"  :
                (i == 1)  ? "weights/weight_L1_N1.mif"  :
                (i == 2)  ? "weights/weight_L1_N2.mif"  :
                (i == 3)  ? "weights/weight_L1_N3.mif"  :
                (i == 4)  ? "weights/weight_L1_N4.mif"  :
                (i == 5)  ? "weights/weight_L1_N5.mif"  :
                (i == 6)  ? "weights/weight_L1_N6.mif"  :
                (i == 7)  ? "weights/weight_L1_N7.mif"  :
                (i == 8)  ? "weights/weight_L1_N8.mif"  :
                (i == 9)  ? "weights/weight_L1_N9.mif"  :
                "weights/default.mif";

            localparam string biasFile = 
                (i == 0)  ? "bias/bias_L1_N0.mif"  :
                (i == 1)  ? "bias/bias_L1_N1.mif"  :
                (i == 2)  ? "bias/bias_L1_N2.mif"  :
                (i == 3)  ? "bias/bias_L1_N3.mif"  :
                (i == 4)  ? "bias/bias_L1_N4.mif"  :
                (i == 5)  ? "bias/bias_L1_N5.mif"  :
                (i == 6)  ? "bias/bias_L1_N6.mif"  :
                (i == 7)  ? "bias/bias_L1_N7.mif"  :
                (i == 8)  ? "bias/bias_L1_N8.mif"  :
                (i == 9)  ? "bias/bias_L1_N9.mif"  :
                "bias/default.mif";

            neuron #(
                .layerNumber(layerNumber),
                .neuronNumber(i),
                .numWeights(numInputs),
                .dataWidth(dataWidth),
                .dataIntWidth(dataIntWidth),
                .dataFracWidth(dataFracWidth),
                .weightWidth(weightWidth),
                .weightIntWidth(weightIntWidth),
                .weightFracWidth(weightFracWidth),
                // File paths for weights and biases
                .weightFile(weightFile),
                .biasFile(biasFile)
            ) Neuron (
                .clk(clk),
                .reset(reset),
                .neuronIn(serializerOut),
                .neuronValid(layerValid),

                .neuronOut(layerOut[(i+1)*dataWidth-1 -: dataWidth]),
                .neuronOutValid(neuronOutValid[i])
            );
        end
    endgenerate

    // Output Logic
    always_comb begin
        layerOutValid = neuronOutValid & counterValid;
    end
endmodule