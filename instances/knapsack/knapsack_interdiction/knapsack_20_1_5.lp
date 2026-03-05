\ Model knapsack_20_1_5
\ LP format - for model browsing. Use MPS format to capture full model detail.
\ Signature: 0xf453f129e35f5108
Minimize
  66.46 x[0] + 102.33 x[1] + 159.67 x[2] + 132.08 x[3] + 141.89 x[4]
   + 172.94 x[5] + 93.89 x[6] + 64.98 x[7] + 108.85 x[8] + 107.54 x[9]
   + 72.98 x[10] + 52.78 x[11] + 61.13 x[12] + 109.76 x[13] + 144.13 x[14]
   + 90.51 x[15] + 53.27 x[16] + 136.61 x[17] + 163.84 x[18] + 160.87 x[19]
   - 337 y[0] - 723 y[1] - 821 y[2] - 924 y[3] - 875 y[4] - 901 y[5]
   - 780 y[6] - 625 y[7] - 925 y[8] - 1039 y[9] - 507 y[10] - 470 y[11]
   - 558 y[12] - 839 y[13] - 1029 y[14] - 481 y[15] - 526 y[16] - 828 y[17]
   - 859 y[18] - 928 y[19]
Subject To
 primal_upper: 237 y[0] + 623 y[1] + 721 y[2] + 824 y[3] + 775 y[4]
   + 801 y[5] + 680 y[6] + 525 y[7] + 825 y[8] + 939 y[9] + 407 y[10]
   + 370 y[11] + 458 y[12] + 739 y[13] + 929 y[14] + 381 y[15] + 426 y[16]
   + 728 y[17] + 759 y[18] + 828 y[19] + 24 z[0] + 62 z[1] + 72 z[2]
   + 82 z[3] + 78 z[4] + 80 z[5] + 68 z[6] + 52 z[7] + 82 z[8] + 94 z[9]
   + 41 z[10] + 37 z[11] + 46 z[12] + 74 z[13] + 93 z[14] + 38 z[15]
   + 43 z[16] + 73 z[17] + 76 z[18] + 83 z[19] <= 2162
 primal_lower0: 931 u[0] + 853 u[1] + 831 u[2] + 950 u[3] + 541 u[4]
   + 863 u[5] + 334 u[6] + 859 u[7] + 799 u[8] + 745 u[9] + 501 u[10]
   + 80 u[11] + 784 u[12] + 433 u[13] + 255 u[14] + 595 u[15] + 128 u[16]
   + 694 u[17] + 181 u[18] + 410 u[19] <= 950
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
