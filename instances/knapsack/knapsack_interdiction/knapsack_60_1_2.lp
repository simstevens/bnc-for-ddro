\ Model knapsack_60_1_2
\ LP format - for model browsing. Use MPS format to capture full model detail.
\ Signature: 0x5769b1bf76edc320
Minimize
  10 x[0] + 10 x[1] + 10 x[2] + 10 x[3] + 10 x[4] + 10 x[5] + 10 x[6]
   + 10 x[7] + 10 x[8] + 10 x[9] + 10 x[10] + 10 x[11] + 10 x[12]
   + 10 x[13] + 10 x[14] + 10 x[15] + 10 x[16] + 10 x[17] + 10 x[18]
   + 10 x[19] + 10 x[20] + 10 x[21] + 10 x[22] + 10 x[23] + 10 x[24]
   + 10 x[25] + 10 x[26] + 10 x[27] + 10 x[28] + 10 x[29] + 10 x[30]
   + 10 x[31] + 10 x[32] + 10 x[33] + 10 x[34] + 10 x[35] + 10 x[36]
   + 10 x[37] + 10 x[38] + 10 x[39] + 10 x[40] + 10 x[41] + 10 x[42]
   + 10 x[43] + 10 x[44] + 10 x[45] + 10 x[46] + 10 x[47] + 10 x[48]
   + 10 x[49] + 10 x[50] + 10 x[51] + 10 x[52] + 10 x[53] + 10 x[54]
   + 10 x[55] + 10 x[56] + 10 x[57] + 10 x[58] + 10 x[59] - 926 y[0]
   - 706 y[1] - 616 y[2] - 712 y[3] - 873 y[4] - 933 y[5] - 881 y[6]
   - 191 y[7] - 94 y[8] - 182 y[9] - 151 y[10] - 583 y[11] - 755 y[12]
   - 454 y[13] - 807 y[14] - 728 y[15] - 334 y[16] - 190 y[17] - 146 y[18]
   - 405 y[19] - 361 y[20] - 691 y[21] - 252 y[22] - 940 y[23] - 41 y[24]
   - 906 y[25] - 200 y[26] - 699 y[27] - 482 y[28] - 764 y[29] - 909 y[30]
   - 428 y[31] - 204 y[32] - 427 y[33] - 771 y[34] - 761 y[35] - 137 y[36]
   - 739 y[37] - 110 y[38] - 631 y[39] - 980 y[40] - 713 y[41] - 809 y[42]
   - 183 y[43] - 930 y[44] - 479 y[45] - 580 y[46] - 128 y[47] - 668 y[48]
   - 397 y[49] - 423 y[50] - 110 y[51] - 259 y[52] - 17 y[53] - 68 y[54]
   - 534 y[55] - 554 y[56] - 496 y[57] - 576 y[58] - 839 y[59]
Subject To
 primal_upper: 141 y[0] + 686 y[1] + 728 y[2] + 349 y[3] + 530 y[4]
   + 635 y[5] + 331 y[6] + 253 y[7] + 851 y[8] + 948 y[9] + 57 y[10]
   + 245 y[11] + 362 y[12] + 246 y[13] + 362 y[14] + 86 y[15] + 523 y[16]
   + 114 y[17] + 471 y[18] + 296 y[19] + 171 y[20] + 673 y[21] + 874 y[22]
   + 128 y[23] + 826 y[24] + 490 y[25] + 216 y[26] + 417 y[27] + 771 y[28]
   + 169 y[29] + 87 y[30] + 717 y[31] + 113 y[32] + 839 y[33] + 658 y[34]
   + 83 y[35] + 458 y[36] + 430 y[37] + 434 y[38] + 560 y[39] + 61 y[40]
   + 419 y[41] + 596 y[42] + 679 y[43] + 197 y[44] + 212 y[45] + 943 y[46]
   + 59 y[47] + 573 y[48] + 50 y[49] + 468 y[50] + 666 y[51] + 971 y[52]
   + 838 y[53] + 230 y[54] + 585 y[55] + 631 y[56] + 628 y[57] + 274 y[58]
   + 740 y[59] + 14 z[0] + 69 z[1] + 73 z[2] + 35 z[3] + 53 z[4] + 64 z[5]
   + 33 z[6] + 25 z[7] + 85 z[8] + 95 z[9] + 6 z[10] + 24 z[11] + 36 z[12]
   + 25 z[13] + 36 z[14] + 9 z[15] + 52 z[16] + 11 z[17] + 47 z[18]
   + 30 z[19] + 17 z[20] + 67 z[21] + 87 z[22] + 13 z[23] + 83 z[24]
   + 49 z[25] + 22 z[26] + 42 z[27] + 77 z[28] + 17 z[29] + 9 z[30]
   + 72 z[31] + 11 z[32] + 84 z[33] + 66 z[34] + 8 z[35] + 46 z[36]
   + 43 z[37] + 43 z[38] + 56 z[39] + 6 z[40] + 42 z[41] + 60 z[42]
   + 68 z[43] + 20 z[44] + 21 z[45] + 94 z[46] + 6 z[47] + 57 z[48]
   + 5 z[49] + 47 z[50] + 67 z[51] + 97 z[52] + 84 z[53] + 23 z[54]
   + 58 z[55] + 63 z[56] + 63 z[57] + 27 z[58] + 74 z[59] <= 9049
 primal_lower0: 923 u[0] + 980 u[1] + 278 u[2] + 931 u[3] + 192 u[4]
   + 489 u[5] + 649 u[6] + 899 u[7] + 737 u[8] + 258 u[9] + 271 u[10]
   + 363 u[11] + 496 u[12] + 724 u[13] + 912 u[14] + 60 u[15] + 481 u[16]
   + 160 u[17] + 757 u[18] + 838 u[19] + 289 u[20] + 911 u[21] + 72 u[22]
   + 846 u[23] + 248 u[24] + 304 u[25] + 454 u[26] + 327 u[27] + 201 u[28]
   + 855 u[29] + 957 u[30] + 339 u[31] + 647 u[32] + 397 u[33] + 256 u[34]
   + 441 u[35] + 856 u[36] + 316 u[37] + 24 u[38] + 630 u[39] + 603 u[40]
   + 73 u[41] + 346 u[42] + 261 u[43] + 507 u[44] + 82 u[45] + 437 u[46]
   + 73 u[47] + 507 u[48] + 808 u[49] + 954 u[50] + 56 u[51] + 225 u[52]
   + 676 u[53] + 4 u[54] + 823 u[55] + 549 u[56] + 258 u[57] + 40 u[58]
   + 170 u[59] <= 9406
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
 interdiction[40]: x[40] + u[40] <= 1
 interdiction[41]: x[41] + u[41] <= 1
 interdiction[42]: x[42] + u[42] <= 1
 interdiction[43]: x[43] + u[43] <= 1
 interdiction[44]: x[44] + u[44] <= 1
 interdiction[45]: x[45] + u[45] <= 1
 interdiction[46]: x[46] + u[46] <= 1
 interdiction[47]: x[47] + u[47] <= 1
 interdiction[48]: x[48] + u[48] <= 1
 interdiction[49]: x[49] + u[49] <= 1
 interdiction[50]: x[50] + u[50] <= 1
 interdiction[51]: x[51] + u[51] <= 1
 interdiction[52]: x[52] + u[52] <= 1
 interdiction[53]: x[53] + u[53] <= 1
 interdiction[54]: x[54] + u[54] <= 1
 interdiction[55]: x[55] + u[55] <= 1
 interdiction[56]: x[56] + u[56] <= 1
 interdiction[57]: x[57] + u[57] <= 1
 interdiction[58]: x[58] + u[58] <= 1
 interdiction[59]: x[59] + u[59] <= 1
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
 mccormick1[40]: - u[40] + z[40] <= 0
 mccormick1[41]: - u[41] + z[41] <= 0
 mccormick1[42]: - u[42] + z[42] <= 0
 mccormick1[43]: - u[43] + z[43] <= 0
 mccormick1[44]: - u[44] + z[44] <= 0
 mccormick1[45]: - u[45] + z[45] <= 0
 mccormick1[46]: - u[46] + z[46] <= 0
 mccormick1[47]: - u[47] + z[47] <= 0
 mccormick1[48]: - u[48] + z[48] <= 0
 mccormick1[49]: - u[49] + z[49] <= 0
 mccormick1[50]: - u[50] + z[50] <= 0
 mccormick1[51]: - u[51] + z[51] <= 0
 mccormick1[52]: - u[52] + z[52] <= 0
 mccormick1[53]: - u[53] + z[53] <= 0
 mccormick1[54]: - u[54] + z[54] <= 0
 mccormick1[55]: - u[55] + z[55] <= 0
 mccormick1[56]: - u[56] + z[56] <= 0
 mccormick1[57]: - u[57] + z[57] <= 0
 mccormick1[58]: - u[58] + z[58] <= 0
 mccormick1[59]: - u[59] + z[59] <= 0
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
 mccormick2[40]: - y[40] + z[40] <= 0
 mccormick2[41]: - y[41] + z[41] <= 0
 mccormick2[42]: - y[42] + z[42] <= 0
 mccormick2[43]: - y[43] + z[43] <= 0
 mccormick2[44]: - y[44] + z[44] <= 0
 mccormick2[45]: - y[45] + z[45] <= 0
 mccormick2[46]: - y[46] + z[46] <= 0
 mccormick2[47]: - y[47] + z[47] <= 0
 mccormick2[48]: - y[48] + z[48] <= 0
 mccormick2[49]: - y[49] + z[49] <= 0
 mccormick2[50]: - y[50] + z[50] <= 0
 mccormick2[51]: - y[51] + z[51] <= 0
 mccormick2[52]: - y[52] + z[52] <= 0
 mccormick2[53]: - y[53] + z[53] <= 0
 mccormick2[54]: - y[54] + z[54] <= 0
 mccormick2[55]: - y[55] + z[55] <= 0
 mccormick2[56]: - y[56] + z[56] <= 0
 mccormick2[57]: - y[57] + z[57] <= 0
 mccormick2[58]: - y[58] + z[58] <= 0
 mccormick2[59]: - y[59] + z[59] <= 0
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
 mccormick3[40]: y[40] + u[40] - z[40] <= 1
 mccormick3[41]: y[41] + u[41] - z[41] <= 1
 mccormick3[42]: y[42] + u[42] - z[42] <= 1
 mccormick3[43]: y[43] + u[43] - z[43] <= 1
 mccormick3[44]: y[44] + u[44] - z[44] <= 1
 mccormick3[45]: y[45] + u[45] - z[45] <= 1
 mccormick3[46]: y[46] + u[46] - z[46] <= 1
 mccormick3[47]: y[47] + u[47] - z[47] <= 1
 mccormick3[48]: y[48] + u[48] - z[48] <= 1
 mccormick3[49]: y[49] + u[49] - z[49] <= 1
 mccormick3[50]: y[50] + u[50] - z[50] <= 1
 mccormick3[51]: y[51] + u[51] - z[51] <= 1
 mccormick3[52]: y[52] + u[52] - z[52] <= 1
 mccormick3[53]: y[53] + u[53] - z[53] <= 1
 mccormick3[54]: y[54] + u[54] - z[54] <= 1
 mccormick3[55]: y[55] + u[55] - z[55] <= 1
 mccormick3[56]: y[56] + u[56] - z[56] <= 1
 mccormick3[57]: y[57] + u[57] - z[57] <= 1
 mccormick3[58]: y[58] + u[58] - z[58] <= 1
 mccormick3[59]: y[59] + u[59] - z[59] <= 1
Bounds
Binaries
 x[0] x[1] x[2] x[3] x[4] x[5] x[6] x[7] x[8] x[9] x[10] x[11] x[12] x[13]
 x[14] x[15] x[16] x[17] x[18] x[19] x[20] x[21] x[22] x[23] x[24] x[25]
 x[26] x[27] x[28] x[29] x[30] x[31] x[32] x[33] x[34] x[35] x[36] x[37]
 x[38] x[39] x[40] x[41] x[42] x[43] x[44] x[45] x[46] x[47] x[48] x[49]
 x[50] x[51] x[52] x[53] x[54] x[55] x[56] x[57] x[58] x[59] y[0] y[1] y[2]
 y[3] y[4] y[5] y[6] y[7] y[8] y[9] y[10] y[11] y[12] y[13] y[14] y[15]
 y[16] y[17] y[18] y[19] y[20] y[21] y[22] y[23] y[24] y[25] y[26] y[27]
 y[28] y[29] y[30] y[31] y[32] y[33] y[34] y[35] y[36] y[37] y[38] y[39]
 y[40] y[41] y[42] y[43] y[44] y[45] y[46] y[47] y[48] y[49] y[50] y[51]
 y[52] y[53] y[54] y[55] y[56] y[57] y[58] y[59] u[0] u[1] u[2] u[3] u[4]
 u[5] u[6] u[7] u[8] u[9] u[10] u[11] u[12] u[13] u[14] u[15] u[16] u[17]
 u[18] u[19] u[20] u[21] u[22] u[23] u[24] u[25] u[26] u[27] u[28] u[29]
 u[30] u[31] u[32] u[33] u[34] u[35] u[36] u[37] u[38] u[39] u[40] u[41]
 u[42] u[43] u[44] u[45] u[46] u[47] u[48] u[49] u[50] u[51] u[52] u[53]
 u[54] u[55] u[56] u[57] u[58] u[59] z[0] z[1] z[2] z[3] z[4] z[5] z[6]
 z[7] z[8] z[9] z[10] z[11] z[12] z[13] z[14] z[15] z[16] z[17] z[18] z[19]
 z[20] z[21] z[22] z[23] z[24] z[25] z[26] z[27] z[28] z[29] z[30] z[31]
 z[32] z[33] z[34] z[35] z[36] z[37] z[38] z[39] z[40] z[41] z[42] z[43]
 z[44] z[45] z[46] z[47] z[48] z[49] z[50] z[51] z[52] z[53] z[54] z[55]
 z[56] z[57] z[58] z[59]
End
