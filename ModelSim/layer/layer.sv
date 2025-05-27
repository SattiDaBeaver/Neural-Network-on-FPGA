module layer #(
    parameter layerNumber = 0, dataWidth = 8, numInputs = 784, numNeurons = 16
) (
    input   logic                               clk,
    input   logic                               reset, 
    input   logic   [dataWidth*numInputs-1:0]   layerIn,
    input   logic                               layerValid,

    output  logic   [dataWidth*numNeurons-1:0]  layerOut,
    output  logic                               layerOutValid
);

    // Parameters
    parameter counterWidth = $clog2(numNeurons+1);

    // Internal Nets
    logic [dataWidth-1:0] serializerOut;
    logic counterValid;

    logic neuronOutValid;

    // Serializer Instance
    inputSerializer #(
        .numInputs(numInputs), 
        .dataWidth(dataWidth), 
        .counterWidth(counterWidth)
    ) U_serializer (
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
            localparam weightFile = 
            (i == 0)  ? "weight_L0_N0.mif"  :
            (i == 1)  ? "weight_L0_N1.mif"  :
            (i == 2)  ? "weight_L0_N2.mif"  :
            (i == 3)  ? "weight_L0_N3.mif"  :
            (i == 4)  ? "weight_L0_N4.mif"  :
            (i == 5)  ? "weight_L0_N5.mif"  :
            (i == 6)  ? "weight_L0_N6.mif"  :
            (i == 7)  ? "weight_L0_N7.mif"  :
            (i == 8)  ? "weight_L0_N8.mif"  :
            (i == 9)  ? "weight_L0_N9.mif"  :
            (i == 10) ? "weight_L0_N10.mif" :
            (i == 11) ? "weight_L0_N11.mif" :
            (i == 12) ? "weight_L0_N12.mif" :
            (i == 13) ? "weight_L0_N13.mif" :
            (i == 14) ? "weight_L0_N14.mif" :
            (i == 15) ? "weight_L0_N15.mif" :
                        "default.mif";

            localparam biasFile = 
            (i == 0)  ? "bias_L0_N0.mif"  :
            (i == 1)  ? "bias_L0_N1.mif"  :
            (i == 2)  ? "bias_L0_N2.mif"  :
            (i == 3)  ? "bias_L0_N3.mif"  :
            (i == 4)  ? "bias_L0_N4.mif"  :
            (i == 5)  ? "bias_L0_N5.mif"  :
            (i == 6)  ? "bias_L0_N6.mif"  :
            (i == 7)  ? "bias_L0_N7.mif"  :
            (i == 8)  ? "bias_L0_N8.mif"  :
            (i == 9)  ? "bias_L0_N9.mif"  :
            (i == 10) ? "bias_L0_N10.mif" :
            (i == 11) ? "bias_L0_N11.mif" :
            (i == 12) ? "bias_L0_N12.mif" :
            (i == 13) ? "bias_L0_N13.mif" :
            (i == 14) ? "bias_L0_N14.mif" :
            (i == 15) ? "bias_L0_N15.mif" :
                        "default.mif";


            neuron #(
                .layerNumber(layerNumber),
                .neuronNumber(i),
                .numWeights(numInputs),
                .dataWidth(dataWidth),
                .weightFile(weightFile),
                .biasFile(biasFile)
            ) Neuron (
                .clk(clk),
                .reset(reset),
                .neuronIn(serializerOut),
                .neuronValid(layerValid),

                .neuronOut(layerOut[(i+1)*dataWidth-1 -: dataWidth]),
                .neuronOutValid(neuronOutValid)
            );
        end
    endgenerate

    // Output Logic
    always_comb begin
        layerOutValid = neuronOutValid & counterValid;
    end
endmodule