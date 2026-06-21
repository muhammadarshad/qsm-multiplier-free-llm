module mac_unit (
    input wire clk,
    input wire reset,
    input wire en,
    input wire [7:0] w_val,
    input wire [7:0] x_val,
    output reg [31:0] acc_out
);

    reg [15:0] prod;
    always @(posedge clk) begin
        if (reset) begin
            prod <= 16'd0;
        end else begin
            prod <= w_val * x_val; // Synthesizes to hardware multiplier block
        end
    end

    reg en_reg;
    always @(posedge clk) begin
        if (reset) begin
            en_reg  <= 1'b0;
            acc_out <= 32'd0;
        end else begin
            en_reg <= en;
            if (en_reg) begin
                acc_out <= acc_out + prod;
            end
        end
    end

endmodule
