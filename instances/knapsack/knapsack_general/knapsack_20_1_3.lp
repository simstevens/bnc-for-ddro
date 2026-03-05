\ Model knapsack_20_1_3
\ LP format - for model browsing. Use MPS format to capture full model detail.
\ Signature: 0x81b32103c6e3a266
Minimize
  - 260 y[0] - 370 y[1] - 440 y[2] - 899 y[3] - 1070 y[4] - 300 y[5]
   - 143 y[6] - 416 y[7] - 1032 y[8] - 830 y[9] - 846 y[10] - 229 y[11]
   - 449 y[12] - 166 y[13] - 504 y[14] - 872 y[15] - 105 y[16] - 939 y[17]
   - 606 y[18] - 271 y[19]
Subject To
 primal_upper: 160 y[0] + 270 y[1] + 340 y[2] + 799 y[3] + 970 y[4]
   + 200 y[5] + 43 y[6] + 316 y[7] + 932 y[8] + 730 y[9] + 746 y[10]
   + 129 y[11] + 349 y[12] + 66 y[13] + 404 y[14] + 772 y[15] + 5 y[16]
   + 839 y[17] + 506 y[18] + 171 y[19] + 16 z[0] + 27 z[1] + 34 z[2]
   + 80 z[3] + 97 z[4] + 20 z[5] + 4 z[6] + 32 z[7] + 93 z[8] + 73 z[9]
   + 75 z[10] + 13 z[11] + 35 z[12] + 7 z[13] + 40 z[14] + 77 z[15]
   + 84 z[17] + 51 z[18] + 17 z[19] <= 2186
 primal_lower0: - 10 y[0] - y[1] - 13 y[2] - 8 y[3] - 14 y[4] - 7 y[5]
   - 33 y[6] - 36 y[7] - 24 y[8] - 11 y[9] - 11 y[10] - 24 y[11] - 6 y[12]
   - 26 y[13] - 23 y[14] - 32 y[15] - 35 y[16] - 7 y[17] - 4 y[18]
   - 44 y[19] + 118 u[0] + 323 u[1] + 55 u[2] + 823 u[3] + 393 u[4]
   + 757 u[5] + 934 u[6] + 957 u[7] + 683 u[8] + 830 u[9] + 596 u[10]
   + 266 u[11] + 372 u[12] + 758 u[13] + 926 u[14] + 84 u[15] + 404 u[16]
   + 651 u[17] + 149 u[18] + 139 u[19] <= 957
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
 y[0] y[1] y[2] y[3] y[4] y[5] y[6] y[7] y[8] y[9] y[10] y[11] y[12] y[13]
 y[14] y[15] y[16] y[17] y[18] y[19] u[0] u[1] u[2] u[3] u[4] u[5] u[6]
 u[7] u[8] u[9] u[10] u[11] u[12] u[13] u[14] u[15] u[16] u[17] u[18] u[19]
 z[0] z[1] z[2] z[3] z[4] z[5] z[6] z[7] z[8] z[9] z[10] z[11] z[12] z[13]
 z[14] z[15] z[16] z[17] z[18] z[19]
End
