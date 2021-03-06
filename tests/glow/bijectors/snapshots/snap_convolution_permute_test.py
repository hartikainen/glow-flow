# -*- coding: utf-8 -*-
# snapshottest: v1 - https://goo.gl/zC4yUc
from __future__ import unicode_literals

from snapshottest import Snapshot


snapshots = Snapshot()

snapshots['TestConvolutionPermute::testForward 1'] = [
    [
        [
            [
                0.4711694121360779,
                -1.0591696500778198,
                -0.10511741042137146
            ],
            [
                0.3510398268699646,
                -0.5476275086402893,
                0.2638520300388336
            ],
            [
                -0.05639129877090454,
                -0.8472033739089966,
                0.22180600464344025
            ],
            [
                0.21365827322006226,
                -0.6015540361404419,
                0.1390717625617981
            ]
        ],
        [
            [
                0.5074745416641235,
                -1.2715398073196411,
                0.11871665716171265
            ],
            [
                0.32578638195991516,
                -0.3426191210746765,
                0.04412208870053291
            ],
            [
                -0.15077081322669983,
                -0.8725491166114807,
                -0.5855400562286377
            ],
            [
                0.40352699160575867,
                -1.069891333580017,
                -0.05621594190597534
            ]
        ],
        [
            [
                0.19868987798690796,
                -0.5066371560096741,
                0.07627873867750168
            ],
            [
                0.37983497977256775,
                -0.8106485605239868,
                -0.5165677666664124
            ],
            [
                0.1068946123123169,
                -1.4940438270568848,
                -0.03375086188316345
            ],
            [
                0.06779754161834717,
                -0.9042289853096008,
                -0.23633860051631927
            ]
        ],
        [
            [
                0.5924555063247681,
                -0.5186610817909241,
                0.029586605727672577
            ],
            [
                0.21867325901985168,
                -1.0660971403121948,
                0.24178023636341095
            ],
            [
                -0.046073317527770996,
                -1.2211809158325195,
                -0.2427540421485901
            ],
            [
                0.6405125856399536,
                -0.801613450050354,
                0.37841707468032837
            ]
        ]
    ]
]

snapshots['TestConvolutionPermute::testInverse 1'] = [
    [
        [
            [
                0.673187255859375,
                -1.1765975952148438,
                0.6716232299804688
            ],
            [
                -0.19711168110370636,
                -1.1104443073272705,
                0.27182871103286743
            ],
            [
                0.6438020467758179,
                -0.7052874565124512,
                0.7233755588531494
            ],
            [
                0.11063506454229355,
                -1.1624598503112793,
                0.4420529305934906
            ]
        ],
        [
            [
                0.6618467569351196,
                -0.34330442547798157,
                0.3380831182003021
            ],
            [
                -0.11619602143764496,
                -0.6221194267272949,
                0.36221978068351746
            ],
            [
                0.03636506199836731,
                -0.7495774030685425,
                0.5309285521507263
            ],
            [
                0.40971073508262634,
                -0.7792198061943054,
                0.5159791707992554
            ]
        ],
        [
            [
                -0.19908757507801056,
                -1.0452792644500732,
                -0.13710856437683105
            ],
            [
                0.35681572556495667,
                -0.6864737272262573,
                -0.032985419034957886
            ],
            [
                -0.024887122213840485,
                -1.1564754247665405,
                0.41258084774017334
            ],
            [
                0.10759911686182022,
                -0.5264546871185303,
                0.8630629181861877
            ]
        ],
        [
            [
                0.7188878655433655,
                -1.0637233257293701,
                -0.3153188228607178
            ],
            [
                0.4830590486526489,
                -1.4463834762573242,
                0.3901904821395874
            ],
            [
                0.6955447793006897,
                -0.4889107942581177,
                0.10120218992233276
            ],
            [
                -0.22853240370750427,
                -1.1604986190795898,
                0.19069638848304749
            ]
        ]
    ]
]

snapshots['TestConvolutionPermute::testForwardLogDetJacobian 1'] = [
    [
        9.998408359024324e-07
    ]
]

snapshots['TestConvolutionPermute::testInverseLogDetJacobian 1'] = [
    [
        -3.5585372870627907e-07
    ]
]
