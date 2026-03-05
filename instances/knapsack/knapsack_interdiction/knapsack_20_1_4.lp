\ Model knapsack_20_1_4
\ LP format - for model browsing. Use MPS format to capture full model detail.
\ Signature: 0xf453f129e35f5108
Minimize
  50.03 x[0] + 101.19 x[1] + 73.6 x[2] + 112.53 x[3] + 67.68 x[4]
   + 167.17 x[5] + 161.3 x[6] + 48.79 x[7] + 121.2 x[8] + 143.02 x[9]
   + 79.97 x[10] + 171.62 x[11] + 28.51 x[12] + 124.13 x[13] + 56.6 x[14]
   + 113.4 x[15] + 27.73 x[16] + 66.63 x[17] + 88.86 x[18] + 26.69 x[19]
   - 280 y[0] - 782 y[1] - 372 y[2] - 631 y[3] - 618 y[4] - 896 y[5]
   - 875 y[6] - 412 y[7] - 796 y[8] - 746 y[9] - 466 y[10] - 1001 y[11]
   - 165 y[12] - 882 y[13] - 332 y[14] - 692 y[15] - 269 y[16] - 599 y[17]
   - 642 y[18] - 235 y[19]
Subject To
 primal_upper: 180 y[0] + 682 y[1] + 272 y[2] + 531 y[3] + 518 y[4]
   + 796 y[5] + 775 y[6] + 312 y[7] + 696 y[8] + 646 y[9] + 366 y[10]
   + 901 y[11] + 65 y[12] + 782 y[13] + 232 y[14] + 592 y[15] + 169 y[16]
   + 499 y[17] + 542 y[18] + 135 y[19] + 18 z[0] + 68 z[1] + 27 z[2]
   + 53 z[3] + 52 z[4] + 80 z[5] + 78 z[6] + 31 z[7] + 70 z[8] + 65 z[9]
   + 37 z[10] + 90 z[11] + 6 z[12] + 78 z[13] + 23 z[14] + 59 z[15]
   + 17 z[16] + 50 z[17] + 54 z[18] + 14 z[19] <= 1938
 primal_lower0: 87 u[0] + 664 u[1] + 912 u[2] + 748 u[3] + 690 u[4]
   + 522 u[5] + 263 u[6] + 338 u[7] + 116 u[8] + 99 u[9] + 997 u[10]
   + 511 u[11] + 213 u[12] + 379 u[13] + 983 u[14] + 377 u[15] + 37 u[16]
   + 616 u[17] + 126 u[18] + 296 u[19] <= 1794
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
