\ Model knapsack_40_1_3
\ LP format - for model browsing. Use MPS format to capture full model detail.
\ Signature: 0xe2fdb6ddddddb322
Minimize
  10 x[0] + 10 x[1] + 10 x[2] + 10 x[3] + 10 x[4] + 10 x[5] + 10 x[6]
   + 10 x[7] + 10 x[8] + 10 x[9] + 10 x[10] + 10 x[11] + 10 x[12]
   + 10 x[13] + 10 x[14] + 10 x[15] + 10 x[16] + 10 x[17] + 10 x[18]
   + 10 x[19] + 10 x[20] + 10 x[21] + 10 x[22] + 10 x[23] + 10 x[24]
   + 10 x[25] + 10 x[26] + 10 x[27] + 10 x[28] + 10 x[29] + 10 x[30]
   + 10 x[31] + 10 x[32] + 10 x[33] + 10 x[34] + 10 x[35] + 10 x[36]
   + 10 x[37] + 10 x[38] + 10 x[39] - 966 y[0] - 418 y[1] - 808 y[2]
   - 808 y[3] - 145 y[4] - 381 y[5] - 761 y[6] - 247 y[7] - 798 y[8]
   - 934 y[9] - 415 y[10] - 943 y[11] - 499 y[12] - 430 y[13] - 359 y[14]
   - 232 y[15] - 566 y[16] - 462 y[17] - 930 y[18] - 677 y[19] - 177 y[20]
   - 91 y[21] - 252 y[22] - 940 y[23] - 177 y[24] - 146 y[25] - 792 y[26]
   - 787 y[27] - 754 y[28] - 652 y[29] - 325 y[30] - 316 y[31] - 972 y[32]
   - 595 y[33] - 387 y[34] - 193 y[35] - 273 y[36] - 483 y[37] - 382 y[38]
   - 871 y[39]
Subject To
 primal_upper: 741 y[0] + 934 y[1] + 504 y[2] + 397 y[3] + 562 y[4]
   + 611 y[5] + 595 y[6] + 853 y[7] + 771 y[8] + 76 y[9] + 409 y[10]
   + 805 y[11] + 634 y[12] + 990 y[13] + 682 y[14] + 878 y[15] + 619 y[16]
   + 810 y[17] + 823 y[18] + 280 y[19] + 571 y[20] + 569 y[21] + 578 y[22]
   + 520 y[23] + 378 y[24] + 810 y[25] + 688 y[26] + 153 y[27] + 939 y[28]
   + 161 y[29] + 927 y[30] + 965 y[31] + 881 y[32] + 327 y[33] + 570 y[34]
   + 827 y[35] + 26 y[36] + 638 y[37] + 170 y[38] + 24 y[39] + 74 z[0]
   + 93 z[1] + 50 z[2] + 40 z[3] + 56 z[4] + 61 z[5] + 60 z[6] + 85 z[7]
   + 77 z[8] + 8 z[9] + 41 z[10] + 80 z[11] + 63 z[12] + 99 z[13]
   + 68 z[14] + 88 z[15] + 62 z[16] + 81 z[17] + 82 z[18] + 28 z[19]
   + 57 z[20] + 57 z[21] + 58 z[22] + 52 z[23] + 38 z[24] + 81 z[25]
   + 69 z[26] + 15 z[27] + 94 z[28] + 16 z[29] + 93 z[30] + 96 z[31]
   + 88 z[32] + 33 z[33] + 57 z[34] + 83 z[35] + 3 z[36] + 64 z[37]
   + 17 z[38] + 2 z[39] <= 5924
 primal_lower0: 803 u[0] + 7 u[1] + 485 u[2] + 526 u[3] + 495 u[4]
   + 765 u[5] + 656 u[6] + 47 u[7] + 168 u[8] + 141 u[9] + 167 u[10]
   + 134 u[11] + 559 u[12] + 783 u[13] + 88 u[14] + 32 u[15] + 936 u[16]
   + 531 u[17] + 828 u[18] + 529 u[19] + 376 u[20] + 195 u[21] + 431 u[22]
   + 954 u[23] + 88 u[24] + 995 u[25] + 37 u[26] + 651 u[27] + 801 u[28]
   + 106 u[29] + 453 u[30] + 879 u[31] + 255 u[32] + 608 u[33] + 71 u[34]
   + 756 u[35] + 639 u[36] + 968 u[37] + 255 u[38] + 778 u[39] <= 4744
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
 interdiction[20]: x[20] + u[20] <= 1
 interdiction[21]: x[21] + u[21] <= 1
 interdiction[22]: x[22] + u[22] <= 1
 interdiction[23]: x[23] + u[23] <= 1
 interdiction[24]: x[24] + u[24] <= 1
 interdiction[25]: x[25] + u[25] <= 1
 interdiction[26]: x[26] + u[26] <= 1
 interdiction[27]: x[27] + u[27] <= 1
 interdiction[28]: x[28] + u[28] <= 1
 interdiction[29]: x[29] + u[29] <= 1
 interdiction[30]: x[30] + u[30] <= 1
 interdiction[31]: x[31] + u[31] <= 1
 interdiction[32]: x[32] + u[32] <= 1
 interdiction[33]: x[33] + u[33] <= 1
 interdiction[34]: x[34] + u[34] <= 1
 interdiction[35]: x[35] + u[35] <= 1
 interdiction[36]: x[36] + u[36] <= 1
 interdiction[37]: x[37] + u[37] <= 1
 interdiction[38]: x[38] + u[38] <= 1
 interdiction[39]: x[39] + u[39] <= 1
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
 mccormick1[20]: - u[20] + z[20] <= 0
 mccormick1[21]: - u[21] + z[21] <= 0
 mccormick1[22]: - u[22] + z[22] <= 0
 mccormick1[23]: - u[23] + z[23] <= 0
 mccormick1[24]: - u[24] + z[24] <= 0
 mccormick1[25]: - u[25] + z[25] <= 0
 mccormick1[26]: - u[26] + z[26] <= 0
 mccormick1[27]: - u[27] + z[27] <= 0
 mccormick1[28]: - u[28] + z[28] <= 0
 mccormick1[29]: - u[29] + z[29] <= 0
 mccormick1[30]: - u[30] + z[30] <= 0
 mccormick1[31]: - u[31] + z[31] <= 0
 mccormick1[32]: - u[32] + z[32] <= 0
 mccormick1[33]: - u[33] + z[33] <= 0
 mccormick1[34]: - u[34] + z[34] <= 0
 mccormick1[35]: - u[35] + z[35] <= 0
 mccormick1[36]: - u[36] + z[36] <= 0
 mccormick1[37]: - u[37] + z[37] <= 0
 mccormick1[38]: - u[38] + z[38] <= 0
 mccormick1[39]: - u[39] + z[39] <= 0
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
 mccormick2[20]: - y[20] + z[20] <= 0
 mccormick2[21]: - y[21] + z[21] <= 0
 mccormick2[22]: - y[22] + z[22] <= 0
 mccormick2[23]: - y[23] + z[23] <= 0
 mccormick2[24]: - y[24] + z[24] <= 0
 mccormick2[25]: - y[25] + z[25] <= 0
 mccormick2[26]: - y[26] + z[26] <= 0
 mccormick2[27]: - y[27] + z[27] <= 0
 mccormick2[28]: - y[28] + z[28] <= 0
 mccormick2[29]: - y[29] + z[29] <= 0
 mccormick2[30]: - y[30] + z[30] <= 0
 mccormick2[31]: - y[31] + z[31] <= 0
 mccormick2[32]: - y[32] + z[32] <= 0
 mccormick2[33]: - y[33] + z[33] <= 0
 mccormick2[34]: - y[34] + z[34] <= 0
 mccormick2[35]: - y[35] + z[35] <= 0
 mccormick2[36]: - y[36] + z[36] <= 0
 mccormick2[37]: - y[37] + z[37] <= 0
 mccormick2[38]: - y[38] + z[38] <= 0
 mccormick2[39]: - y[39] + z[39] <= 0
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
 mccormick3[20]: y[20] + u[20] - z[20] <= 1
 mccormick3[21]: y[21] + u[21] - z[21] <= 1
 mccormick3[22]: y[22] + u[22] - z[22] <= 1
 mccormick3[23]: y[23] + u[23] - z[23] <= 1
 mccormick3[24]: y[24] + u[24] - z[24] <= 1
 mccormick3[25]: y[25] + u[25] - z[25] <= 1
 mccormick3[26]: y[26] + u[26] - z[26] <= 1
 mccormick3[27]: y[27] + u[27] - z[27] <= 1
 mccormick3[28]: y[28] + u[28] - z[28] <= 1
 mccormick3[29]: y[29] + u[29] - z[29] <= 1
 mccormick3[30]: y[30] + u[30] - z[30] <= 1
 mccormick3[31]: y[31] + u[31] - z[31] <= 1
 mccormick3[32]: y[32] + u[32] - z[32] <= 1
 mccormick3[33]: y[33] + u[33] - z[33] <= 1
 mccormick3[34]: y[34] + u[34] - z[34] <= 1
 mccormick3[35]: y[35] + u[35] - z[35] <= 1
 mccormick3[36]: y[36] + u[36] - z[36] <= 1
 mccormick3[37]: y[37] + u[37] - z[37] <= 1
 mccormick3[38]: y[38] + u[38] - z[38] <= 1
 mccormick3[39]: y[39] + u[39] - z[39] <= 1
Bounds
Binaries
 x[0] x[1] x[2] x[3] x[4] x[5] x[6] x[7] x[8] x[9] x[10] x[11] x[12] x[13]
 x[14] x[15] x[16] x[17] x[18] x[19] x[20] x[21] x[22] x[23] x[24] x[25]
 x[26] x[27] x[28] x[29] x[30] x[31] x[32] x[33] x[34] x[35] x[36] x[37]
 x[38] x[39] y[0] y[1] y[2] y[3] y[4] y[5] y[6] y[7] y[8] y[9] y[10] y[11]
 y[12] y[13] y[14] y[15] y[16] y[17] y[18] y[19] y[20] y[21] y[22] y[23]
 y[24] y[25] y[26] y[27] y[28] y[29] y[30] y[31] y[32] y[33] y[34] y[35]
 y[36] y[37] y[38] y[39] u[0] u[1] u[2] u[3] u[4] u[5] u[6] u[7] u[8] u[9]
 u[10] u[11] u[12] u[13] u[14] u[15] u[16] u[17] u[18] u[19] u[20] u[21]
 u[22] u[23] u[24] u[25] u[26] u[27] u[28] u[29] u[30] u[31] u[32] u[33]
 u[34] u[35] u[36] u[37] u[38] u[39] z[0] z[1] z[2] z[3] z[4] z[5] z[6]
 z[7] z[8] z[9] z[10] z[11] z[12] z[13] z[14] z[15] z[16] z[17] z[18] z[19]
 z[20] z[21] z[22] z[23] z[24] z[25] z[26] z[27] z[28] z[29] z[30] z[31]
 z[32] z[33] z[34] z[35] z[36] z[37] z[38] z[39]
End
