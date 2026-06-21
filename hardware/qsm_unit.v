module qsm_unit (
    input wire clk,
    input wire reset,
    input wire en,
    input wire [7:0] w_val,
    input wire [7:0] x_val,
    output reg [31:0] acc_out
);

    // Two independent ROMs to allow single-cycle lookups for sum and diff
    reg [17:0] sq_rom_sum  [0:510];
    reg [17:0] sq_rom_diff [0:510];

    initial begin
        $readmemh("sq_rom.hex", sq_rom_sum);
        $readmemh("sq_rom.hex", sq_rom_diff);
    end

    // Compute sum and absolute difference
    wire [8:0] s_val;
    wire [7:0] d_val;

    assign s_val = w_val + x_val;
    assign d_val = (w_val >= x_val) ? (w_val - x_val) : (x_val - w_val);

    // ROM lookups (synchronous reads for standard RAM synthesis)
    reg [17:0] sq_sum;
    reg [17:0] sq_diff;

    always @(posedge clk) begin
        if (reset) begin
            sq_sum  <= 18'd0;
            sq_diff <= 18'd0;
        end else begin
            sq_sum  <= sq_rom_sum[s_val];
            sq_diff <= sq_rom_diff[d_val];
        end
    end

    // QSM product subtraction and division by 4
    wire [17:0] qsm_prod;
    assign qsm_prod = (sq_sum - sq_diff) >> 2;

    // Pipeline register to accumulate
    reg en_reg;
    always @(posedge clk) begin
        if (reset) begin
            en_reg  <= 1'b0;
            acc_out <= 32'd0;
        end else begin
            en_reg <= en;
            if (en_reg) begin
                acc_out <= acc_out + qsm_prod;
            end
        end
    end

endmodule
