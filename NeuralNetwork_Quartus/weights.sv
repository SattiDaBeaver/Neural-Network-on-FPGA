`define PRETRAINED
module weights #(
    parameter numWeights = 16, neuronNumber = 0, layerNumber = 1, addressWidth = 4, dataWidth = 16, weightFile = "w_l0_n0.mif"
) (
    input   logic                       clk,
    input   logic                       readEn,
    input   logic                       writeEn,
    input   logic   [addressWidth-1:0]  addr,
    input   logic   [dataWidth-1:0]     dataIn,
    output  logic   [dataWidth-1:0]     dataOut
);

    // Force Quartus to use M9K block RAM on MAX 10
    (* ramstyle = "M9K" *)  
    logic   [dataWidth-1:0]   mem   [0:numWeights-1];     // data width (bits) x depth (number of weights)


    `ifdef PRETRAINED
        initial begin
            $readmemb(weightFile, mem);
        end
    `else
        always_ff @ (posedge clk) begin
            if (writeEn) begin
                mem[addr] <= dataIn;
            end
        end
    `endif
    
    always_comb begin
        if (readEn) begin
            dataOut = mem[addr];
        end
        else dataOut = 0;
    end

endmodule: weights