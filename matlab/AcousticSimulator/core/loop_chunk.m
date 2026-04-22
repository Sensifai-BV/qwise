function c = loop_chunk(wav, ptr, N)
%LOOP_CHUNK  Read N samples from a circular buffer starting at ptr (1-based).
    L   = length(wav);
    idx = mod(ptr - 1 + (0:N-1)', L) + 1;
    c   = wav(idx);
end
