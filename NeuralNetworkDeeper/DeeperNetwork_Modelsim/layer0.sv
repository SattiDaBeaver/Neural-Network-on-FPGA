module layer0 #(
    parameter   layerNumber = 0, numInputs = 784, numNeurons = 16,
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
        .counterWidth(counterWidth),
        .isFirstLayer(1) // This is the first layer
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
                (i == 0)  ? "weights/weight_L0_N0.mif"  :
                (i == 1)  ? "weights/weight_L0_N1.mif"  :
                (i == 2)  ? "weights/weight_L0_N2.mif"  :
                (i == 3)  ? "weights/weight_L0_N3.mif"  :
                (i == 4)  ? "weights/weight_L0_N4.mif"  :
                (i == 5)  ? "weights/weight_L0_N5.mif"  :
                (i == 6)  ? "weights/weight_L0_N6.mif"  :
                (i == 7)  ? "weights/weight_L0_N7.mif"  :
                (i == 8)  ? "weights/weight_L0_N8.mif"  :
                (i == 9)  ? "weights/weight_L0_N9.mif"  :
                (i == 10) ? "weights/weight_L0_N10.mif" :
                (i == 11) ? "weights/weight_L0_N11.mif" :
                (i == 12) ? "weights/weight_L0_N12.mif" :
                (i == 13) ? "weights/weight_L0_N13.mif" :
                (i == 14) ? "weights/weight_L0_N14.mif" :
                (i == 15) ? "weights/weight_L0_N15.mif" :
                "weights/default.mif";


            localparam string biasFile = 
                (i == 0)  ? "bias/bias_L0_N0.mif"  :
                (i == 1)  ? "bias/bias_L0_N1.mif"  :
                (i == 2)  ? "bias/bias_L0_N2.mif"  :
                (i == 3)  ? "bias/bias_L0_N3.mif"  :
                (i == 4)  ? "bias/bias_L0_N4.mif"  :
                (i == 5)  ? "bias/bias_L0_N5.mif"  :
                (i == 6)  ? "bias/bias_L0_N6.mif"  :
                (i == 7)  ? "bias/bias_L0_N7.mif"  :
                (i == 8)  ? "bias/bias_L0_N8.mif"  :
                (i == 9)  ? "bias/bias_L0_N9.mif"  :
                (i == 10) ? "bias/bias_L0_N10.mif" :
                (i == 11) ? "bias/bias_L0_N11.mif" :
                (i == 12) ? "bias/bias_L0_N12.mif" :
                (i == 13) ? "bias/bias_L0_N13.mif" :
                (i == 14) ? "bias/bias_L0_N14.mif" :
                (i == 15) ? "bias/bias_L0_N15.mif" :
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

                .neuronOut(layerOut[i*dataWidth +: dataWidth]),
                .neuronOutValid(neuronOutValid[i])
            );
        end
    endgenerate

    // Output Logic
    always_comb begin
        layerOutValid = neuronOutValid & counterValid;
    end
endmodule