''' 
Pretty ASCII prints

'''

import os
import numpy as np
from colorama import init as colorama_init, Fore, Back, Style

# Initialize colors
colorama_init()

# ==========================================================================
#                            COMMAND WINDOW
# ========================================================================== 

def line(text=None, thickness=1, char=None):
  '''
  Pretty ASCII line

  Print a line spanning the whole command line.

  By default it is a single line (─) but other characters can be used: 
    Double line: ═
    Trople line: ≡

  '''

  # Terminal width
  tw = os.get_terminal_size().columns

  # Thickness
  if char is None:
    match thickness:
      case 1: char = '─'
      case 2: char = '═'
      case 3: char = '≡'

  if text is None or text=='':
    S = char*tw

  else:
    S = char*3 + ' ' + text + ' '
    S += char*(tw-len(S))

  # Display
  print(S)

# ==========================================================================
#                            MATRIX DIPLAY
# ========================================================================== 

def matrix(M, maxrow=20, maxcol=20, chsep=' ', halign='right', escchar='░',
           highlight=None, highColor=Fore.RED, title=None,
           cTrue='1', cFalse='.',
           precision=None):
    ''' 
    Pretty ASCII print of a matrix
    '''

    # === Title ============================================================

    if title is not None:
      line(title, 1)
      print('')

    # === Checks ===========================================================
    
    if not isinstance(M, np.ndarray):
      M = np.ndarray(M)

    # === Colors ===========================================================

    # Hidden parts
    escchar = Fore.YELLOW + escchar + Style.RESET_ALL

    # Highlights
    if highlight is not None:
      Hlg = highlight[0:maxrow, 0:maxcol]

    # === Max digits =======================================================

    # Subpanel
    Sub = M[0:maxrow, 0:maxcol]

    # Booleans
    if np.issubdtype(Sub.dtype, np.bool_):
      ms = max(len(cTrue), len(cFalse))

    # Integers
    elif np.issubdtype(Sub.dtype, np.integer):

      ms = int(np.ceil(np.max(1*(Sub<0) + np.log10(np.abs(Sub)))))

    # Floats
    elif np.issubdtype(Sub.dtype, np.floating):

      msi = int(np.ceil(np.max(1*(Sub<0) + np.maximum(1, np.log10(np.abs(Sub)))))) + 1
      prec = max(4-msi, 0) if precision is None else precision
      ms = msi + prec - (prec==0)

    else:
      raise TypeError(f'{__name__} > matrix : dtype {Sub.dtype!r} is not recognized')

    # === Column headers ===================================================

    # Row/column header max symbol
    rms = int(max(1,np.ceil(np.log10(Sub.shape[0]))))
    cms = int(max(1,np.ceil(np.log10(Sub.shape[1]))))

    # Row or column format?
    rowf = cms<=ms

    if rowf:

      # --- Index line

      hdr = [' '*(rms+2) + Fore.BLACK]

      for j in range(Sub.shape[1]):
        match halign:
          case 'r' | 'right':
            hdr[0] += f' {j:>{ms}d}'
          case 'c' | 'center':
            hdr[0] += f' {j:^{ms}d}'
          case 'l' | 'left':
            hdr[0] += f' {j:<{ms}d}'

      hdr[0] += Style.RESET_ALL

    else:

      # Prepare rows
      hdr = [' '*(rms+3)+Fore.BLACK for i in range(cms)]

      for j in range(Sub.shape[1]):

        if j==0:

          for k in range(cms):
            h = ' ' if k < cms-1 else '0'
            hdr[k] += f'{h:{ms+1}s}'
        
        else:

          for k in range(cms):

            if j<10**(cms-k-1):
              h = ' '

            else:
              h = f'{j // 10**(cms-k-1) % 10:d}'

            hdr[k] += f'{h:{ms+1}s}'

      hdr[-1] += Style.RESET_ALL

    # --- Deco line header

    r = ' '*(rms+1) + '┌' + ' '*((ms+1)*Sub.shape[1]+1) 
    if M.shape[1]>maxcol: r += escchar*2 + ' '
    r += '┐'
    hdr.append(r)

    # --- Deco line footer

    r = ' '*(rms+1) + '└' + ' '*((ms+1)*Sub.shape[1]+1)
    if M.shape[1]>maxcol: 
      r += '   ' if M.shape[0]>maxrow else escchar*2 + ' '
    r += '┘'
    ftr = [r]

    # Print header
    for r in hdr: print(r)

    # === Rows =============================================================

    for i, row in enumerate(M):

      # Rows to skip
      if i>maxrow-1:
        r = ' '*(rms+1) + '│' + escchar*((ms+1)*Sub.shape[1]+1) 
        if M.shape[1]>maxcol: r += escchar*2 + ' '
        r += '│'
        print(r)
        break

      # Row header
      print(Fore.BLACK + f'{i:>{rms}d}' + Style.RESET_ALL + ' │', end='')

      for j, a in enumerate(row):

        # Columns to skip
        if j>maxcol-1:
          print(' ' + escchar*2, end='')
          break

        # --- Cell content

        # Boolean
        if np.issubdtype(a, np.bool_):
          s = cTrue if a else cFalse

        # Integers
        elif np.issubdtype(a, np.integer):
          match halign:
            case 'r' | 'right':
              s = f'{a:>{ms}d}'
            case 'c' | 'center':
              s = f'{a:^{ms}d}'
            case 'l' | 'left':
              s = f'{a:<{ms}d}'

        # Floats
        else:
          match halign:
            case 'r' | 'right':
              s = f'{a:>{ms}.{prec}f}'
            case 'c' | 'center':
              s = f'{a:^{ms}.{prec}f}'
            case 'l' | 'left':
              s = f'{a:<{ms}.{prec}f}'

        if highlight is None or not Hlg[i,j]:
          print(' ' + s, end='')
        else:
          print(' ' + highColor + s + Style.RESET_ALL, end='')

      print(' │')

    # === Footer ===========================================================

    for r in ftr: print(r)

    print('')