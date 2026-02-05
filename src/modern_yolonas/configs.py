"""Hardcoded architecture parameters for YOLO-NAS S/M/L variants.

Replaces YAML+OmegaConf configs from super-gradients.
Values extracted from the official arch_params YAML files.
"""

YOLO_NAS_S = {
    "backbone": {
        "stem_out_channels": 48,
        "stages": [
            {"out_channels": 96, "num_blocks": 2, "hidden_channels": 32, "concat_intermediates": False},
            {"out_channels": 192, "num_blocks": 3, "hidden_channels": 64, "concat_intermediates": False},
            {"out_channels": 384, "num_blocks": 5, "hidden_channels": 96, "concat_intermediates": False},
            {"out_channels": 768, "num_blocks": 2, "hidden_channels": 192, "concat_intermediates": False},
        ],
        "spp_output_channels": 768,
        "spp_k": (5, 9, 13),
    },
    "neck": {
        "neck1_config": {
            "out_channels": 192,
            "num_blocks": 2,
            "hidden_channels": 64,
            "width_mult": 1,
            "depth_mult": 1,
            "reduce_channels": True,
        },
        "neck2_config": {
            "out_channels": 96,
            "num_blocks": 2,
            "hidden_channels": 48,
            "width_mult": 1,
            "depth_mult": 1,
            "reduce_channels": True,
        },
        "neck3_config": {
            "out_channels": 192,
            "num_blocks": 2,
            "hidden_channels": 64,
            "width_mult": 1,
            "depth_mult": 1,
        },
        "neck4_config": {
            "out_channels": 384,
            "num_blocks": 2,
            "hidden_channels": 64,
            "width_mult": 1,
            "depth_mult": 1,
        },
    },
    "heads": {
        "reg_max": 16,
        "heads_list": [
            {"inter_channels": 128, "width_mult": 0.5, "first_conv_group_size": 0, "stride": 8},
            {"inter_channels": 256, "width_mult": 0.5, "first_conv_group_size": 0, "stride": 16},
            {"inter_channels": 512, "width_mult": 0.5, "first_conv_group_size": 0, "stride": 32},
        ],
    },
}

YOLO_NAS_M = {
    "backbone": {
        "stem_out_channels": 48,
        "stages": [
            {"out_channels": 96, "num_blocks": 2, "hidden_channels": 64, "concat_intermediates": True},
            {"out_channels": 192, "num_blocks": 3, "hidden_channels": 128, "concat_intermediates": True},
            {"out_channels": 384, "num_blocks": 5, "hidden_channels": 256, "concat_intermediates": True},
            {"out_channels": 768, "num_blocks": 2, "hidden_channels": 384, "concat_intermediates": False},
        ],
        "spp_output_channels": 768,
        "spp_k": (5, 9, 13),
    },
    "neck": {
        "neck1_config": {
            "out_channels": 192,
            "num_blocks": 2,
            "hidden_channels": 192,
            "width_mult": 1,
            "depth_mult": 1,
            "reduce_channels": True,
        },
        "neck2_config": {
            "out_channels": 96,
            "num_blocks": 3,
            "hidden_channels": 64,
            "width_mult": 1,
            "depth_mult": 1,
            "reduce_channels": True,
        },
        "neck3_config": {
            "out_channels": 192,
            "num_blocks": 2,
            "hidden_channels": 192,
            "width_mult": 1,
            "depth_mult": 1,
        },
        "neck4_config": {
            "out_channels": 384,
            "num_blocks": 3,
            "hidden_channels": 256,
            "width_mult": 1,
            "depth_mult": 1,
        },
    },
    "heads": {
        "reg_max": 16,
        "heads_list": [
            {"inter_channels": 128, "width_mult": 0.75, "first_conv_group_size": 0, "stride": 8},
            {"inter_channels": 256, "width_mult": 0.75, "first_conv_group_size": 0, "stride": 16},
            {"inter_channels": 512, "width_mult": 0.75, "first_conv_group_size": 0, "stride": 32},
        ],
    },
}

YOLO_NAS_L = {
    "backbone": {
        "stem_out_channels": 48,
        "stages": [
            {"out_channels": 96, "num_blocks": 2, "hidden_channels": 96, "concat_intermediates": True},
            {"out_channels": 192, "num_blocks": 3, "hidden_channels": 128, "concat_intermediates": True},
            {"out_channels": 384, "num_blocks": 5, "hidden_channels": 256, "concat_intermediates": True},
            {"out_channels": 768, "num_blocks": 2, "hidden_channels": 512, "concat_intermediates": True},
        ],
        "spp_output_channels": 768,
        "spp_k": (5, 9, 13),
    },
    "neck": {
        "neck1_config": {
            "out_channels": 192,
            "num_blocks": 4,
            "hidden_channels": 128,
            "width_mult": 1,
            "depth_mult": 1,
            "reduce_channels": True,
        },
        "neck2_config": {
            "out_channels": 96,
            "num_blocks": 4,
            "hidden_channels": 128,
            "width_mult": 1,
            "depth_mult": 1,
            "reduce_channels": True,
        },
        "neck3_config": {
            "out_channels": 192,
            "num_blocks": 4,
            "hidden_channels": 128,
            "width_mult": 1,
            "depth_mult": 1,
        },
        "neck4_config": {
            "out_channels": 384,
            "num_blocks": 4,
            "hidden_channels": 256,
            "width_mult": 1,
            "depth_mult": 1,
        },
    },
    "heads": {
        "reg_max": 16,
        "heads_list": [
            {"inter_channels": 128, "width_mult": 1, "first_conv_group_size": 0, "stride": 8},
            {"inter_channels": 256, "width_mult": 1, "first_conv_group_size": 0, "stride": 16},
            {"inter_channels": 512, "width_mult": 1, "first_conv_group_size": 0, "stride": 32},
        ],
    },
}

CONFIGS = {
    "yolo_nas_s": YOLO_NAS_S,
    "yolo_nas_m": YOLO_NAS_M,
    "yolo_nas_l": YOLO_NAS_L,
}
