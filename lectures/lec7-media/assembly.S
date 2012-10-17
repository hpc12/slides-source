   4:	c7 45 f4 05 00 00 00 movl   $0x5,-0xc(%rbp)
   b:	c7 45 f8 11 00 00 00 movl   $0x11,-0x8(%rbp)
  12:	8b 45 f4             mov    -0xc(%rbp),%eax
  15:	0f af 45 f8          imul   -0x8(%rbp),%eax
  19:	89 45 fc             mov    %eax,-0x4(%rbp)
  1c:	8b 45 fc             mov    -0x4(%rbp),%eax
