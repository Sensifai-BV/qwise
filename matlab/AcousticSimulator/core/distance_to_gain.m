function g = distance_to_gain(d)
%DISTANCE_TO_GAIN  Piecewise outdoor speech-level mapping for the
%   Q-WiSE perChannel demo path.
%
%   g = distance_to_gain(d) returns the speech level the simulated
%   laptop microphone (mic-1) should observe when the human talker is
%   D metres from the static drone. The mapping is the perceptual
%   table the user specified for the outdoor demo:
%
%       0.10 m ≤ d ≤ 1.0  m → 0.95
%       1.0  m  < d ≤ 2.0  m → 0.75
%       2.0  m  < d ≤ 4.0  m → 0.50
%       4.0  m  < d ≤ 6.0  m → 0.30
%       6.0  m  < d ≤ 8.0  m → 0.10
%       8.0  m  < d            → 0.00   (includes the 8–10 m band and
%                                         anything beyond, clamped to 0)
%       d < 0.10 m              → 0.95   (clamp very-close into the
%                                         in-range top band — no boost)
%
%   The function is intentionally NOT the physical 1/d law: it is a
%   demo-grade level mapping for the perChannel mixer mode. The
%   research-grade `physical` mode keeps the 1/d gains in
%   build_geometry.gains_speech.
%
%   d may be a scalar, vector, or matrix; g preserves the shape of d.
%   See also: SourceMixer (perChannel branch), build_geometry.

    d = double(d);
    g = zeros(size(d));

    g(d <= 1.0)              = 0.95;
    g(d > 1.0  & d <= 2.0)   = 0.75;
    g(d > 2.0  & d <= 4.0)   = 0.50;
    g(d > 4.0  & d <= 6.0)   = 0.30;
    g(d > 6.0  & d <= 8.0)   = 0.10;
    g(d > 8.0)               = 0.00;
end
