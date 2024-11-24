function isAlpha(cCheck) {
    return ((('a'<=cCheck) && (cCheck<='z')) || (('A'<=cCheck) && (cCheck<='Z'))) 
}

function naturallyLength(s, n) {
    const old_n = n
    if (s.length > n) {
      while (n > 0 && isAlpha(s[n])) --n;
    }
    if (n == 0) {
        n = old_n
    }
    return n
}

function naturallySlice(s, n) {
    return s.slice(0, naturallyLength(s, n))
}

function naturallyShorten(s, n) {
    const m = naturallyLength(s, n)
    if (m < n) {
        return s.slice(0, m) + '...'
    } else {
        return s
    }
}

export default { naturallyLength, naturallySlice, naturallyShorten }