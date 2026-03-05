\ Model knapsack_20_1_2
\ LP format - for model browsing. Use MPS format to capture full model detail.
\ Signature: 0xf453f129e35f5108
Minimize
  25.97 x[0] + 108.1 x[1] + 57.45 x[2] + 69.17 x[3] + 173.65 x[4]
   + 73.06 x[5] + 23.8 x[6] + 78.93 x[7] + 121.72 x[8] + 91.08 x[9]
   + 114.17 x[10] + 136.62 x[11] + 75.75 x[12] + 136.15 x[13] + 17.88 x[14]
   + 72.72 x[15] + 60.92 x[16] + 88.08 x[17] + 73.73 x[18] + 66.09 x[19]
   - 185 y[0] - 563 y[1] - 565 y[2] - 348 y[3] - 1019 y[4] - 405 y[5]
   - 132 y[6] - 657 y[7] - 669 y[8] - 551 y[9] - 995 y[10] - 734 y[11]
   - 526 y[12] - 791 y[13] - 165 y[14] - 433 y[15] - 606 y[16] - 828 y[17]
   - 667 y[18] - 632 y[19]
Subject To
 primal_upper: 85 y[0] + 463 y[1] + 465 y[2] + 248 y[3] + 919 y[4]
   + 305 y[5] + 32 y[6] + 557 y[7] + 569 y[8] + 451 y[9] + 895 y[10]
   + 634 y[11] + 426 y[12] + 691 y[13] + 65 y[14] + 333 y[15] + 506 y[16]
   + 728 y[17] + 567 y[18] + 532 y[19] + 8 z[0] + 46 z[1] + 46 z[2]
   + 25 z[3] + 92 z[4] + 30 z[5] + 3 z[6] + 56 z[7] + 57 z[8] + 45 z[9]
   + 90 z[10] + 63 z[11] + 43 z[12] + 69 z[13] + 6 z[14] + 33 z[15]
   + 51 z[16] + 73 z[17] + 57 z[18] + 53 z[19] <= 3157
 primal_lower0: 664 u[0] + 686 u[1] + 244 u[2] + 55 u[3] + 986 u[4]
   + 496 u[5] + 315 u[6] + 860 u[7] + 308 u[8] + 722 u[9] + 482 u[10]
   + 521 u[11] + 157 u[12] + 858 u[13] + 604 u[14] + 964 u[15] + 277 u[16]
   + 87 u[17] + 994 u[18] + 339 u[19] <= 994
 interdiction[0]: x[0] + u[0] <= 1
 interdiction[1]: x[1] + u[1] <= 1
 interdiction[2]: x[2] + u[2] <= 1
 interdiction[3]: x[3] + u[3] <= 1
 interdiction[4]: x[4] + u[4] <= 1
 interdiction[5]: x[5] + u[5] <= 1
 interdiction[6]: x[6] + u[6] <= 1
 interdiction[7]: x[7] + u[7] <= 1
 interdiction[8]: x[8] + u[8] <= 1
 interdiction[9]: x[9] + u[9] <= 1
 interdiction[10]: x[10] + u[10] <= 1
 interdiction[11]: x[11] + u[11] <= 1
 interdiction[12]: x[12] + u[12] <= 1
 interdiction[13]: x[13] + u[13] <= 1
 interdiction[14]: x[14] + u[14] <= 1
 interdiction[15]: x[15] + u[15] <= 1
 interdiction[16]: x[16] + u[16] <= 1
 interdiction[17]: x[17] + u[17] <= 1
 interdiction[18]: x[18] + u[18] <= 1
 interdiction[19]: x[19] + u[19] <= 1
 mccormick1[0]: - u[0] + z[0] <= 0
 mccormick1[1]: - u[1] + z[1] <= 0
 mccormick1[2]: - u[2] + z[2] <= 0
 mccormick1[3]: - u[3] + z[3] <= 0
 mccormick1[4]: - u[4] + z[4] <= 0
 mccormick1[5]: - u[5] + z[5] <= 0
 mccormick1[6]: - u[6] + z[6] <= 0
 mccormick1[7]: - u[7] + z[7] <= 0
 mccormick1[8]: - u[8] + z[8] <= 0
 mccormick1[9]: - u[9] + z[9] <= 0
 mccormick1[10]: - u[10] + z[10] <= 0
 mccormick1[11]: - u[11] + z[11] <= 0
 mccormick1[12]: - u[12] + z[12] <= 0
 mccormick1[13]: - u[13] + z[13] <= 0
 mccormick1[14]: - u[14] + z[14] <= 0
 mccormick1[15]: - u[15] + z[15] <= 0
 mccormick1[16]: - u[16] + z[16] <= 0
 mccormick1[17]: - u[17] + z[17] <= 0
 mccormick1[18]: - u[18] + z[18] <= 0
 mccormick1[19]: - u[19] + z[19] <= 0
 mccormick2[0]: - y[0] + z[0] <= 0
 mccormick2[1]: - y[1] + z[1] <= 0
 mccormick2[2]: - y[2] + z[2] <= 0
 mccormick2[3]: - y[3] + z[3] <= 0
 mccormick2[4]: - y[4] + z[4] <= 0
 mccormick2[5]: - y[5] + z[5] <= 0
 mccormick2[6]: - y[6] + z[6] <= 0
 mccormick2[7]: - y[7] + z[7] <= 0
 mccormick2[8]: - y[8] + z[8] <= 0
 mccormick2[9]: - y[9] + z[9] <= 0
 mccormick2[10]: - y[10] + z[10] <= 0
 mccormick2[11]: - y[11] + z[11] <= 0
 mccormick2[12]: - y[12] + z[12] <= 0
 mccormick2[13]: - y[13] + z[13] <= 0
 mccormick2[14]: - y[14] + z[14] <= 0
 mccormick2[15]: - y[15] + z[15] <= 0
 mccormick2[16]: - y[16] + z[16] <= 0
 mccormick2[17]: - y[17] + z[17] <= 0
 mccormick2[18]: - y[18] + z[18] <= 0
 mccormick2[19]: - y[19] + z[19] <= 0
 mccormick3[0]: y[0] + u[0] - z[0] <= 1
 mccormick3[1]: y[1] + u[1] - z[1] <= 1
 mccormick3[2]: y[2] + u[2] - z[2] <= 1
 mccormick3[3]: y[3] + u[3] - z[3] <= 1
 mccormick3[4]: y[4] + u[4] - z[4] <= 1
 mccormick3[5]: y[5] + u[5] - z[5] <= 1
 mccormick3[6]: y[6] + u[6] - z[6] <= 1
 mccormick3[7]: y[7] + u[7] - z[7] <= 1
 mccormick3[8]: y[8] + u[8] - z[8] <= 1
 mccormick3[9]: y[9] + u[9] - z[9] <= 1
 mccormick3[10]: y[10] + u[10] - z[10] <= 1
 mccormick3[11]: y[11] + u[11] - z[11] <= 1
 mccormick3[12]: y[12] + u[12] - z[12] <= 1
 mccormick3[13]: y[13] + u[13] - z[13] <= 1
 mccormick3[14]: y[14] + u[14] - z[14] <= 1
 mccormick3[15]: y[15] + u[15] - z[15] <= 1
 mccormick3[16]: y[16] + u[16] - z[16] <= 1
 mccormick3[17]: y[17] + u[17] - z[17] <= 1
 mccormick3[18]: y[18] + u[18] - z[18] <= 1
 mccormick3[19]: y[19] + u[19] - z[19] <= 1
Bounds
Binaries
 x[0] x[1] x[2] x[3] x[4] x[5] x[6] x[7] x[8] x[9] x[10] x[11] x[12] x[13]
 x[14] x[15] x[16] x[17] x[18] x[19] y[0] y[1] y[2] y[3] y[4] y[5] y[6]
 y[7] y[8] y[9] y[10] y[11] y[12] y[13] y[14] y[15] y[16] y[17] y[18] y[19]
 u[0] u[1] u[2] u[3] u[4] u[5] u[6] u[7] u[8] u[9] u[10] u[11] u[12] u[13]
 u[14] u[15] u[16] u[17] u[18] u[19] z[0] z[1] z[2] z[3] z[4] z[5] z[6]
 z[7] z[8] z[9] z[10] z[11] z[12] z[13] z[14] z[15] z[16] z[17] z[18] z[19]
End
