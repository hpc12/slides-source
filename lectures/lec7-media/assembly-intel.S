   4:	c7 45 f4 05 00 00 00 	mov    DWORD PTR [rbp-0xc],0x5
   b:	c7 45 f8 11 00 00 00 	mov    DWORD PTR [rbp-0x8],0x11
  12:	8b 45 f4             	mov    eax,DWORD PTR [rbp-0xc]
  15:	0f af 45 f8          	imul   eax,DWORD PTR [rbp-0x8]
  19:	89 45 fc             	mov    DWORD PTR [rbp-0x4],eax
  1c:	8b 45 fc             	mov    eax,DWORD PTR [rbp-0x4]
