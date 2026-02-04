# 文件名: llava/model/multimodal_encoder/reltr_models_adapt/flops_utils.py
import math

class RelTRFLOPsCounter:
    def __init__(self, 
                 d_model=256, 
                 d_ffn=2048, 
                 num_enc_layers=3, 
                 num_dec_layers=3,
                 num_entities=100, 
                 num_triplets=200,
                 num_classes=151,
                 num_rel_classes=51,
                 input_proj_dim=768,
                 topk_selection=10):
        self.d = d_model
        self.d_ff = d_ffn
        self.L_enc = num_enc_layers
        self.L_dec = num_dec_layers
        self.Ne = num_entities
        self.Nt = num_triplets
        self.num_classes = num_classes
        self.num_rel_classes = num_rel_classes
        self.Cin_proj = input_proj_dim
        self.topk = topk_selection
        self.S_so = 2 * num_triplets

    def _format_g(self, x):
        return f"{x / 1e9:.4f} GFLOPs"
    
    def _format_sub(self, x):
        return f"{x / 1e9:.4f}"

    def _flops_linear(self, N, din, dout):
        return 2 * N * din * dout

    def _flops_conv(self, B, Cin, Cout, H, W, k):
        return 2 * B * H * W * Cout * (Cin * k * k)

    def _flops_sa(self, S):
        return 8 * S * self.d**2 + 4 * S**2 * self.d

    def _flops_ca(self, Q, K):
        return 4 * (Q + K) * self.d**2 + 4 * Q * K * self.d

    def _flops_ffn(self, S):
        return 4 * S * self.d * self.d_ff

    def calculate(self, feat_h, feat_w):
        K_mem = feat_h * feat_w
        S_enc_attn = K_mem + 1
        S_enc_ffn = K_mem

        m_proj = self._flops_linear(K_mem, self.Cin_proj, self.d) + \
                 self._flops_linear(1, self.Cin_proj, self.d)

        m_enc_layer = self._flops_sa(S_enc_attn) + self._flops_ffn(S_enc_ffn)
        m_enc = self.L_enc * m_enc_layer

        dec_A = self._flops_sa(self.Ne) + \
                self._flops_ca(self.Ne, K_mem) + \
                self._flops_ffn(self.Ne)
        dec_B = self._flops_sa(self.S_so)
        dec_C = self._flops_ca(self.Nt, K_mem) + \
                self._flops_ca(self.Nt, self.Ne) + \
                self._flops_ffn(self.Nt)
        dec_D = dec_C
        
        m_dec = self.L_dec * (dec_A + dec_B + dec_C + dec_D)

        B_conv = self.L_dec * self.Nt
        mask_c1 = self._flops_conv(B_conv, 2, 64, 16, 16, 3)
        mask_c2 = self._flops_conv(B_conv, 64, 32, 8, 8, 3)
        m_mask_conv = mask_c1 + mask_c2

        m_mask_fc = self._flops_linear(B_conv, 2048, 128)

        h_cls = self._flops_linear(self.L_dec * self.Ne, self.d, self.num_classes + 1) + \
                2 * self._flops_linear(self.L_dec * self.Nt, self.d, self.num_classes + 1)

        def bbox_ops(N):
            return self._flops_linear(N, self.d, self.d) * 2 + self._flops_linear(N, self.d, 4)
        h_bbox = bbox_ops(self.L_dec * self.Ne) + 2 * bbox_ops(self.L_dec * self.Nt)

        h_rel = self._flops_linear(self.L_dec * self.Nt, 640, self.d) + \
                self._flops_linear(self.L_dec * self.Nt, self.d, self.num_rel_classes + 1)

        total_logits = self.Nt * (self.num_rel_classes + 2 * self.num_classes)
        m_sel_softmax = total_logits * 4
        m_sel_dot = self._flops_linear(self.topk, 768, 1)

        total = m_proj + m_enc + m_dec + m_mask_conv + m_mask_fc + h_cls + h_bbox + h_rel + m_sel_softmax + m_sel_dot

        w = 26
        print(f"\n{'='*20} Auto FLOPs Report {'='*20}")
        print(f"{'Original:':<{w}} ({feat_h}*{feat_w})")
        print(f"{'[1] Projection:':<{w}} {self._format_g(m_proj)}")
        print(f"{'[2] Encoder:':<{w}} {self._format_g(m_enc)}  (L={S_enc_attn})")
        
        print(f"{'[3] Decoder:':<{w}} {self._format_g(m_dec)}")
        print(f"    - A Entity:            {self._format_sub(dec_A)}")
        print(f"    - B Coupled:           {self._format_sub(dec_B)}")
        print(f"    - C/D Sub/Obj:         {self._format_sub(dec_C)}")
        
        print(f"{'[4] so_mask_conv:':<{w}} {self._format_g(m_mask_conv)}")
        print(f"{'[5] so_mask_fc:':<{w}} {self._format_g(m_mask_fc)}")
        print(f"{'[6] Class heads:':<{w}} {self._format_g(h_cls)}")
        print(f"{'[7] BBox heads:':<{w}} {self._format_g(h_bbox)}")
        print(f"{'[8] Rel head:':<{w}} {self._format_g(h_rel)}")
        
        print(f"{'[9] Selection (Total):':<{w}} {self._format_g(m_sel_softmax + m_sel_dot)}")
        print(f"    - Softmax:             {m_sel_softmax/1e9:.6f} G")
        print(f"    - Dot (topk={self.topk}):        {m_sel_dot/1e9:.6f} G")
        
        print("-" * 45)
        print(f"{'TOTAL:':<{w}} {self._format_g(total)}")
        print(f"{'='*60}\n")
        
        return total