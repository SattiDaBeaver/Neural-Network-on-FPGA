`timescale 1ns / 1ps
// `define PRETRAINED

module testbench ( );

	parameter CLOCK_PERIOD = 10;

	logic CLOCK_50;
    logic [0:0] KEY;
    logic [6:0] HEX0;

	// Neuron Test
	logic 				reset;
	logic [15:0] 		neuronIn;
	logic [15:0] 		neuronOut;
	logic 				neuronValid;
	logic 				neuronOutValid;

    // Serializer Test
    parameter numInputs = 784;
    parameter dataWidth = 16;
    parameter counterWidth = $clog2(numInputs);
    logic [numInputs*dataWidth-1:0] inputData;
    logic [dataWidth-1:0] serializerOut;
    logic [counterWidth-1:0] counterOut;
    logic counterValid;


    // Weight Memory
    logic [15:0]       weightMem [0:numInputs-1];

    // Graham Ball
    logic [784*16-1:0] inputMem [0:0];

    initial begin
        $readmemh("Inputs/input_0_q8_8.txt", inputMem);
    end

    // initial begin
    //     $readmemb("weights/weight_L0_N0.mif", weightMem);
    //     $display("Loaded weights:");
    //     for (int i = 0; i < 16; i++)
    //         $display("weightMem[%0d] = %b", i, weightMem[i]);
    // end

	initial begin
        CLOCK_50 <= 1'b0;
	end // initial
	always @ (*)
	begin : Clock_Generator
		#((CLOCK_PERIOD) / 2) CLOCK_50 <= ~CLOCK_50;
	end
	
	initial begin
        reset <= 1'b0;
        neuronValid <= 1'b0;
        #10
        reset <= 1'b1;
		#10 
		reset <= 1'b0;
		inputData <= inputMem[0];
        #20
		neuronValid <= 1'b1;
        
	end // initial

    always_ff @(posedge CLOCK_50) begin
        if (neuronOutValid)
            $display("Time=%t | neuronOut = %h", $time, neuronOut);
    end

	
	neuron #(
        .layerNumber(0),
        .neuronNumber(0),
        .numWeights(numInputs),
        .dataWidth(dataWidth),
        .dataIntWidth(8),
        .dataFracWidth(8),
        .weightWidth(16),
        .weightIntWidth(8),
        .weightFracWidth(8),
        .biasFile("bias/bias_L0_N1.mif"),
        .weightFile("weights/weight_L0_N1.mif")
    ) U1 (
        .clk(CLOCK_50),
        .reset(reset),
        .neuronIn(serializerOut),
        .neuronValid(neuronValid),
        .weightValid(),
        .weightWriteEn(),
        .biasWriteEn(),
        .weightData(32'h0),
        .biasData(32'h0),
        .config_layer_number(32'h0),
        .config_neuron_number(32'h0),
        .neuronOut(neuronOut),
        .neuronOutValid(neuronOutValid)
    );

     // Serializer Instance
    inputSerializer #(
        .numInputs(numInputs), 
        .dataWidth(dataWidth), 
        .counterWidth(counterWidth)
    ) serializer (
        .clk(CLOCK_50),
        .reset(reset),
        .enable(neuronValid),
        .serializerIn(inputData),
        .counterValid(counterValid),
        .serializerOut(serializerOut)
    );

endmodule
