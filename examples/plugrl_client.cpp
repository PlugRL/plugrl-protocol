// A PlugRL env client in C++ that depends on nothing but libstdc++ and a
// TCP socket.
//
// The Python client in this directory already showed the protocol can be
// spoken without PlugRL or numpy. This one closes the remaining gap: it is a
// different language, and it builds on a machine with no msgpack library, no
// WebSocket library, and no OpenSSL - so SHA-1, base64, WebSocket framing and
// the msgpack subset the protocol needs are all here, written against the
// specifications.
//
// That is the situation a robot's onboard controller is actually in. If this
// works, "a ROS node could be an env client" stops being a claim and becomes
// a demonstration.
//
// Build:  g++ -std=c++17 -O2 -o plugrl_client plugrl_client.cpp
// Run:    ./plugrl_client 127.0.0.1 8123 20

#include <arpa/inet.h>
#include <netdb.h>
#include <netinet/in.h>
#include <netinet/tcp.h>
#include <sys/socket.h>
#include <unistd.h>

#include <algorithm>
#include <chrono>
#include <cstdint>
#include <cstring>
#include <iomanip>
#include <iostream>
#include <map>
#include <memory>
#include <numeric>
#include <random>
#include <stdexcept>
#include <string>
#include <vector>

namespace {

// ---------------------------------------------------------------- SHA-1
// RFC 3174. Needed only to verify the server's Sec-WebSocket-Accept.

class Sha1 {
 public:
  Sha1() { reset(); }

  void update(const uint8_t* data, size_t len) {
    for (size_t i = 0; i < len; ++i) {
      buffer_[buffer_len_++] = data[i];
      if (buffer_len_ == 64) {
        transform(buffer_);
        total_ += 64;
        buffer_len_ = 0;
      }
    }
  }

  std::vector<uint8_t> digest() {
    uint64_t total_bits = (total_ + buffer_len_) * 8;
    uint8_t pad = 0x80;
    update(&pad, 1);
    uint8_t zero = 0x00;
    while (buffer_len_ != 56) update(&zero, 1);
    uint8_t len_be[8];
    for (int i = 0; i < 8; ++i) len_be[7 - i] = (total_bits >> (8 * i)) & 0xff;
    update(len_be, 8);

    std::vector<uint8_t> out(20);
    for (int i = 0; i < 5; ++i) {
      out[i * 4 + 0] = (h_[i] >> 24) & 0xff;
      out[i * 4 + 1] = (h_[i] >> 16) & 0xff;
      out[i * 4 + 2] = (h_[i] >> 8) & 0xff;
      out[i * 4 + 3] = h_[i] & 0xff;
    }
    return out;
  }

 private:
  void reset() {
    h_[0] = 0x67452301; h_[1] = 0xEFCDAB89; h_[2] = 0x98BADCFE;
    h_[3] = 0x10325476; h_[4] = 0xC3D2E1F0;
    buffer_len_ = 0; total_ = 0;
  }

  static uint32_t rol(uint32_t v, int b) { return (v << b) | (v >> (32 - b)); }

  void transform(const uint8_t block[64]) {
    uint32_t w[80];
    for (int i = 0; i < 16; ++i)
      w[i] = (block[i * 4] << 24) | (block[i * 4 + 1] << 16) |
             (block[i * 4 + 2] << 8) | block[i * 4 + 3];
    for (int i = 16; i < 80; ++i)
      w[i] = rol(w[i - 3] ^ w[i - 8] ^ w[i - 14] ^ w[i - 16], 1);

    uint32_t a = h_[0], b = h_[1], c = h_[2], d = h_[3], e = h_[4];
    for (int i = 0; i < 80; ++i) {
      uint32_t f, k;
      if (i < 20)      { f = (b & c) | (~b & d);          k = 0x5A827999; }
      else if (i < 40) { f = b ^ c ^ d;                   k = 0x6ED9EBA1; }
      else if (i < 60) { f = (b & c) | (b & d) | (c & d); k = 0x8F1BBCDC; }
      else             { f = b ^ c ^ d;                   k = 0xCA62C1D6; }
      uint32_t t = rol(a, 5) + f + e + k + w[i];
      e = d; d = c; c = rol(b, 30); b = a; a = t;
    }
    h_[0] += a; h_[1] += b; h_[2] += c; h_[3] += d; h_[4] += e;
  }

  uint32_t h_[5];
  uint8_t buffer_[64];
  size_t buffer_len_;
  uint64_t total_;
};

std::string base64_encode(const uint8_t* data, size_t len) {
  static const char* tbl =
      "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/";
  std::string out;
  for (size_t i = 0; i < len; i += 3) {
    uint32_t n = data[i] << 16;
    if (i + 1 < len) n |= data[i + 1] << 8;
    if (i + 2 < len) n |= data[i + 2];
    out += tbl[(n >> 18) & 63];
    out += tbl[(n >> 12) & 63];
    out += (i + 1 < len) ? tbl[(n >> 6) & 63] : '=';
    out += (i + 2 < len) ? tbl[n & 63] : '=';
  }
  return out;
}

// ---------------------------------------------------------------- msgpack
//
// Only what the protocol uses: maps, strings, binary, arrays, integers,
// floats and booleans. No extension types - the wire format does not use any.

class Packer {
 public:
  void map(size_t n) {
    if (n < 16) put_u8(0x80 | static_cast<uint8_t>(n));
    else if (n < 65536) { put_u8(0xde); put_be16(static_cast<uint16_t>(n)); }
    else { put_u8(0xdf); put_be32(static_cast<uint32_t>(n)); }
  }

  void array(size_t n) {
    if (n < 16) put_u8(0x90 | static_cast<uint8_t>(n));
    else if (n < 65536) { put_u8(0xdc); put_be16(static_cast<uint16_t>(n)); }
    else { put_u8(0xdd); put_be32(static_cast<uint32_t>(n)); }
  }

  void str(const std::string& s) {
    size_t n = s.size();
    if (n < 32) put_u8(0xa0 | static_cast<uint8_t>(n));
    else if (n < 256) { put_u8(0xd9); put_u8(static_cast<uint8_t>(n)); }
    else if (n < 65536) { put_u8(0xda); put_be16(static_cast<uint16_t>(n)); }
    else { put_u8(0xdb); put_be32(static_cast<uint32_t>(n)); }
    buf_.append(s);
  }

  void bin(const uint8_t* data, size_t n) {
    if (n < 256) { put_u8(0xc4); put_u8(static_cast<uint8_t>(n)); }
    else if (n < 65536) { put_u8(0xc5); put_be16(static_cast<uint16_t>(n)); }
    else { put_u8(0xc6); put_be32(static_cast<uint32_t>(n)); }
    buf_.append(reinterpret_cast<const char*>(data), n);
  }

  void bin(const std::string& s) {
    bin(reinterpret_cast<const uint8_t*>(s.data()), s.size());
  }

  void integer(int64_t v) {
    if (v >= 0 && v < 128) { put_u8(static_cast<uint8_t>(v)); return; }
    if (v < 0 && v >= -32) { put_u8(static_cast<uint8_t>(0xe0 | (v + 32))); return; }
    put_u8(0xd3);
    for (int i = 7; i >= 0; --i) put_u8((static_cast<uint64_t>(v) >> (8 * i)) & 0xff);
  }

  void boolean(bool v) { put_u8(v ? 0xc3 : 0xc2); }

  const std::string& data() const { return buf_; }
  void clear() { buf_.clear(); }

 private:
  void put_u8(uint8_t v) { buf_.push_back(static_cast<char>(v)); }
  void put_be16(uint16_t v) { put_u8(v >> 8); put_u8(v & 0xff); }
  void put_be32(uint32_t v) {
    put_u8(v >> 24); put_u8((v >> 16) & 0xff); put_u8((v >> 8) & 0xff); put_u8(v & 0xff);
  }
  std::string buf_;
};

struct Value;
using ValuePtr = std::shared_ptr<Value>;

struct Value {
  enum class Kind { Nil, Bool, Int, UInt, Float, Str, Bin, Array, Map };
  Kind kind = Kind::Nil;
  bool b = false;
  int64_t i = 0;
  uint64_t u = 0;
  double f = 0;
  std::string s;              // Str and Bin both land here
  std::vector<ValuePtr> arr;
  std::vector<std::pair<ValuePtr, ValuePtr>> map;

  const ValuePtr find(const std::string& key) const {
    for (const auto& kv : map)
      if ((kv.first->kind == Kind::Str || kv.first->kind == Kind::Bin) &&
          kv.first->s == key)
        return kv.second;
    return nullptr;
  }
};

class Unpacker {
 public:
  Unpacker(const uint8_t* data, size_t len) : p_(data), end_(data + len) {}

  ValuePtr parse() {
    auto v = std::make_shared<Value>();
    uint8_t t = take_u8();

    if (t <= 0x7f) { v->kind = Value::Kind::UInt; v->u = t; return v; }
    if (t >= 0xe0) { v->kind = Value::Kind::Int;
                     v->i = static_cast<int8_t>(t); return v; }
    if ((t & 0xf0) == 0x80) return parse_map(v, t & 0x0f);
    if ((t & 0xf0) == 0x90) return parse_array(v, t & 0x0f);
    if ((t & 0xe0) == 0xa0) return parse_str(v, t & 0x1f);

    switch (t) {
      case 0xc0: v->kind = Value::Kind::Nil; return v;
      case 0xc2: v->kind = Value::Kind::Bool; v->b = false; return v;
      case 0xc3: v->kind = Value::Kind::Bool; v->b = true; return v;
      case 0xc4: return parse_bin(v, take_u8());
      case 0xc5: return parse_bin(v, take_be16());
      case 0xc6: return parse_bin(v, take_be32());
      case 0xca: { v->kind = Value::Kind::Float;
                   uint32_t r = take_be32(); float g;
                   std::memcpy(&g, &r, 4); v->f = g; return v; }
      case 0xcb: { v->kind = Value::Kind::Float;
                   uint64_t r = take_be64(); double g;
                   std::memcpy(&g, &r, 8); v->f = g; return v; }
      case 0xcc: v->kind = Value::Kind::UInt; v->u = take_u8(); return v;
      case 0xcd: v->kind = Value::Kind::UInt; v->u = take_be16(); return v;
      case 0xce: v->kind = Value::Kind::UInt; v->u = take_be32(); return v;
      case 0xcf: v->kind = Value::Kind::UInt; v->u = take_be64(); return v;
      case 0xd0: v->kind = Value::Kind::Int;
                 v->i = static_cast<int8_t>(take_u8()); return v;
      case 0xd1: v->kind = Value::Kind::Int;
                 v->i = static_cast<int16_t>(take_be16()); return v;
      case 0xd2: v->kind = Value::Kind::Int;
                 v->i = static_cast<int32_t>(take_be32()); return v;
      case 0xd3: v->kind = Value::Kind::Int;
                 v->i = static_cast<int64_t>(take_be64()); return v;
      case 0xd9: return parse_str(v, take_u8());
      case 0xda: return parse_str(v, take_be16());
      case 0xdb: return parse_str(v, take_be32());
      case 0xdc: return parse_array(v, take_be16());
      case 0xdd: return parse_array(v, take_be32());
      case 0xde: return parse_map(v, take_be16());
      case 0xdf: return parse_map(v, take_be32());
      default:
        throw std::runtime_error("unsupported msgpack type 0x" +
                                 std::to_string(static_cast<int>(t)));
    }
  }

 private:
  ValuePtr parse_str(ValuePtr v, size_t n) {
    v->kind = Value::Kind::Str; v->s.assign(take(n), n); return v;
  }
  ValuePtr parse_bin(ValuePtr v, size_t n) {
    v->kind = Value::Kind::Bin; v->s.assign(take(n), n); return v;
  }
  ValuePtr parse_array(ValuePtr v, size_t n) {
    v->kind = Value::Kind::Array;
    for (size_t k = 0; k < n; ++k) v->arr.push_back(parse());
    return v;
  }
  ValuePtr parse_map(ValuePtr v, size_t n) {
    v->kind = Value::Kind::Map;
    for (size_t k = 0; k < n; ++k) {
      auto key = parse();
      auto val = parse();
      v->map.emplace_back(key, val);
    }
    return v;
  }

  const char* take(size_t n) {
    if (p_ + n > end_) throw std::runtime_error("msgpack: truncated");
    const char* r = reinterpret_cast<const char*>(p_);
    p_ += n;
    return r;
  }
  uint8_t take_u8() { return static_cast<uint8_t>(*take(1)); }
  uint16_t take_be16() { uint16_t a = take_u8(); return (a << 8) | take_u8(); }
  uint32_t take_be32() { uint32_t a = take_be16(); return (a << 16) | take_be16(); }
  uint64_t take_be64() { uint64_t a = take_be32(); return (a << 32) | take_be32(); }

  const uint8_t* p_;
  const uint8_t* end_;
};

// ---------------------------------------------------------------- ndarray
//
// {b"__ndarray__": true, b"data": <bin>, b"dtype": "<f4", b"shape": [...]}
//
// dtype is a numpy typestr: byte order, kind, item size. The only piece of
// numpy vocabulary in the protocol, and the reason this function exists.

void pack_ndarray(Packer& p, const std::string& raw, const std::string& dtype,
                  const std::vector<int64_t>& shape) {
  p.map(4);
  p.bin(std::string("__ndarray__")); p.boolean(true);
  p.bin(std::string("data"));        p.bin(raw);
  p.bin(std::string("dtype"));       p.str(dtype);
  p.bin(std::string("shape"));
  p.array(shape.size());
  for (int64_t d : shape) p.integer(d);
}

std::string pack_f8(const std::vector<double>& v) {
  std::string out(v.size() * 8, '\0');
  std::memcpy(&out[0], v.data(), out.size());  // x86 is little-endian, "<f8"
  return out;
}

std::string pack_i8(const std::vector<int64_t>& v) {
  std::string out(v.size() * 8, '\0');
  std::memcpy(&out[0], v.data(), out.size());
  return out;
}

// ---------------------------------------------------------------- websocket

class WebSocket {
 public:
  void connect(const std::string& host, int port) {
    addrinfo hints{}, *res = nullptr;
    hints.ai_family = AF_INET;
    hints.ai_socktype = SOCK_STREAM;
    if (getaddrinfo(host.c_str(), std::to_string(port).c_str(), &hints, &res) != 0)
      throw std::runtime_error("cannot resolve " + host);
    fd_ = ::socket(res->ai_family, res->ai_socktype, res->ai_protocol);
    if (fd_ < 0 || ::connect(fd_, res->ai_addr, res->ai_addrlen) != 0) {
      freeaddrinfo(res);
      throw std::runtime_error("cannot connect to " + host);
    }
    freeaddrinfo(res);
    int one = 1;
    setsockopt(fd_, IPPROTO_TCP, TCP_NODELAY, &one, sizeof(one));
    handshake(host, port);
  }

  ~WebSocket() { if (fd_ >= 0) ::close(fd_); }

  void send_binary(const std::string& payload) {
    std::string frame;
    frame.push_back(static_cast<char>(0x82));  // FIN + binary opcode
    size_t n = payload.size();
    uint8_t mask_bit = 0x80;                   // clients must mask
    if (n < 126) {
      frame.push_back(static_cast<char>(mask_bit | n));
    } else if (n < 65536) {
      frame.push_back(static_cast<char>(mask_bit | 126));
      frame.push_back(static_cast<char>((n >> 8) & 0xff));
      frame.push_back(static_cast<char>(n & 0xff));
    } else {
      frame.push_back(static_cast<char>(mask_bit | 127));
      for (int i = 7; i >= 0; --i)
        frame.push_back(static_cast<char>((n >> (8 * i)) & 0xff));
    }
    uint8_t key[4];
    for (int i = 0; i < 4; ++i) key[i] = static_cast<uint8_t>(rng_() & 0xff);
    frame.append(reinterpret_cast<char*>(key), 4);
    size_t off = frame.size();
    frame.append(payload);
    for (size_t i = 0; i < n; ++i) frame[off + i] ^= key[i % 4];
    write_all(frame.data(), frame.size());
  }

  std::string recv_message() {
    std::string message;
    for (;;) {
      uint8_t h[2];
      read_all(h, 2);
      bool fin = h[0] & 0x80;
      uint8_t opcode = h[0] & 0x0f;
      bool masked = h[1] & 0x80;
      uint64_t len = h[1] & 0x7f;
      if (len == 126) {
        uint8_t e[2]; read_all(e, 2);
        len = (static_cast<uint64_t>(e[0]) << 8) | e[1];
      } else if (len == 127) {
        uint8_t e[8]; read_all(e, 8);
        len = 0;
        for (int i = 0; i < 8; ++i) len = (len << 8) | e[i];
      }
      uint8_t key[4] = {0, 0, 0, 0};
      if (masked) read_all(key, 4);

      std::string chunk(len, '\0');
      if (len) read_all(reinterpret_cast<uint8_t*>(&chunk[0]), len);
      if (masked)
        for (uint64_t i = 0; i < len; ++i) chunk[i] ^= key[i % 4];

      if (opcode == 0x8) throw std::runtime_error("server closed the connection");
      if (opcode == 0x9) { send_pong(chunk); continue; }
      if (opcode == 0xa) continue;

      message += chunk;
      if (fin) return message;
    }
  }

 private:
  void handshake(const std::string& host, int port) {
    uint8_t nonce[16];
    for (int i = 0; i < 16; ++i) nonce[i] = static_cast<uint8_t>(rng_() & 0xff);
    std::string key = base64_encode(nonce, 16);

    std::string req =
        "GET / HTTP/1.1\r\n"
        "Host: " + host + ":" + std::to_string(port) + "\r\n"
        "Upgrade: websocket\r\n"
        "Connection: Upgrade\r\n"
        "Sec-WebSocket-Key: " + key + "\r\n"
        "Sec-WebSocket-Version: 13\r\n\r\n";
    write_all(req.data(), req.size());

    std::string resp;
    while (resp.find("\r\n\r\n") == std::string::npos) {
      char c;
      read_all(reinterpret_cast<uint8_t*>(&c), 1);
      resp.push_back(c);
      if (resp.size() > 8192) throw std::runtime_error("handshake response too long");
    }
    if (resp.find("101") == std::string::npos)
      throw std::runtime_error("server refused the upgrade:\n" + resp);

    // Verify Sec-WebSocket-Accept: base64(sha1(key + GUID)).
    const std::string guid = "258EAFA5-E914-47DA-95CA-C5AB0DC85B11";
    Sha1 sha;
    std::string concat = key + guid;
    sha.update(reinterpret_cast<const uint8_t*>(concat.data()), concat.size());
    auto d = sha.digest();
    std::string expect = base64_encode(d.data(), d.size());
    if (resp.find(expect) == std::string::npos)
      throw std::runtime_error("Sec-WebSocket-Accept mismatch; expected " + expect);
  }

  void send_pong(const std::string& payload) {
    std::string frame;
    frame.push_back(static_cast<char>(0x8a));
    frame.push_back(static_cast<char>(0x80 | payload.size()));
    uint8_t key[4] = {0, 0, 0, 0};
    frame.append(reinterpret_cast<char*>(key), 4);
    frame.append(payload);
    write_all(frame.data(), frame.size());
  }

  void write_all(const char* data, size_t n) {
    size_t sent = 0;
    while (sent < n) {
      ssize_t k = ::send(fd_, data + sent, n - sent, 0);
      if (k <= 0) throw std::runtime_error("send failed");
      sent += static_cast<size_t>(k);
    }
  }

  void read_all(uint8_t* data, size_t n) {
    size_t got = 0;
    while (got < n) {
      ssize_t k = ::recv(fd_, data + got, n - got, 0);
      if (k <= 0) throw std::runtime_error("connection closed while reading");
      got += static_cast<size_t>(k);
    }
  }

  int fd_ = -1;
  std::mt19937 rng_{std::random_device{}()};
};

// ---------------------------------------------------------------- client

uint32_t g_seed = 12345;
double next_float() {
  g_seed = (1103515245u * g_seed + 12345u) & 0x7fffffff;
  return static_cast<double>(g_seed % 10000) / 10000.0;
}

// Image geometry is configurable so the cost of the boundary can be measured
// against payload size rather than guessed at.
int g_img_size = 224;
int g_cameras = 2;

void pack_observation(Packer& p, int batch) {
  p.map(3);

  p.str("images");
  p.map(g_cameras);
  static const char* names[4] = {"base", "wrist", "left", "right"};
  for (int k = 0; k < g_cameras; ++k) {
    // The second camera is the usual half-resolution wrist view.
    int side = (k == 1) ? g_img_size / 2 : g_img_size;
    p.str(names[k % 4]);
    size_t n = static_cast<size_t>(batch) * side * side * 3;
    std::string raw(n, '\0');
    for (size_t i = 0; i < n; ++i) raw[i] = static_cast<char>(i & 0xff);
    pack_ndarray(p, raw, "|u1", {batch, side, side, 3});
  }

  p.str("states");
  p.map(2);
  {
    std::vector<double> s(static_cast<size_t>(batch) * 10);
    for (auto& x : s) x = next_float();
    p.str("robot_state");
    pack_ndarray(p, pack_f8(s), "<f8", {batch, 10});
  }
  {
    std::vector<double> j(static_cast<size_t>(batch) * 5);
    for (auto& x : j) x = next_float();
    p.str("joint_angles");
    pack_ndarray(p, pack_f8(j), "<f8", {batch, 5});
  }

  p.str("text");
  p.array(batch);
  for (int i = 0; i < batch; ++i) p.str("do something");
}

int run(const std::string& host, int port, int steps, int batch) {
  WebSocket ws;
  std::cout << "connecting to ws://" << host << ":" << port << "\n";
  ws.connect(host, port);

  auto meta_raw = ws.recv_message();
  Unpacker meta_up(reinterpret_cast<const uint8_t*>(meta_raw.data()), meta_raw.size());
  auto meta = meta_up.parse();
  auto mt = meta->find("message_type");
  if (!mt || mt->s != "metadata") {
    std::cerr << "expected metadata, got " << (mt ? mt->s : "<nothing>") << "\n";
    return 1;
  }
  std::cout << "handshake ok, metadata received\n";

  std::vector<int64_t> env_idx(batch);
  for (int i = 0; i < batch; ++i) env_idx[i] = i;

  using clock = std::chrono::steady_clock;
  auto ms = [](clock::duration d) {
    return std::chrono::duration<double, std::milli>(d).count();
  };
  std::vector<double> pack_ms, rtt_ms, unpack_ms;
  size_t infer_bytes = 0;

  for (int step = 0; step < steps; ++step) {
    std::vector<int64_t> step_ids(batch, step);

    auto t0 = clock::now();
    Packer p;
    p.map(4);
    p.str("message_type"); p.str("infer");
    p.str("data");         pack_observation(p, batch);
    p.str("env_indices");  pack_ndarray(p, pack_i8(env_idx), "<i8", {batch});
    p.str("step_ids");     pack_ndarray(p, pack_i8(step_ids), "<i8", {batch});
    auto t1 = clock::now();

    infer_bytes = p.data().size();
    ws.send_binary(p.data());
    auto reply = ws.recv_message();
    auto t2 = clock::now();

    Unpacker up(reinterpret_cast<const uint8_t*>(reply.data()), reply.size());
    auto msg = up.parse();
    auto t3 = clock::now();

    // Skip the first few: the first exchange carries connection warm-up.
    if (step >= 3) {
      pack_ms.push_back(ms(t1 - t0));
      rtt_ms.push_back(ms(t2 - t1));
      unpack_ms.push_back(ms(t3 - t2));
    }
    auto type = msg->find("message_type");
    if (!type || type->s != "action") {
      std::cerr << "expected action, got " << (type ? type->s : "<nothing>") << "\n";
      return 1;
    }
    auto data = msg->find("data");
    auto action = data ? data->find("action") : nullptr;
    if (!action) { std::cerr << "no action field\n"; return 1; }

    auto dtype = action->find("dtype");
    auto shape = action->find("shape");
    auto blob = action->find("data");
    if (!dtype || !shape || !blob) { std::cerr << "malformed ndarray\n"; return 1; }

    std::vector<int64_t> dims;
    for (const auto& d : shape->arr)
      dims.push_back(d->kind == Value::Kind::Int ? d->i
                                                 : static_cast<int64_t>(d->u));
    for (int64_t d : dims)
      if (d == 0) { std::cerr << "server returned an empty action\n"; return 1; }

    if (step == 0) {
      std::cout << "action: dtype=" << dtype->s << " shape=[";
      for (size_t i = 0; i < dims.size(); ++i)
        std::cout << dims[i] << (i + 1 < dims.size() ? ", " : "");
      std::cout << "]\n        first values:";
      // "<f4" is float32, little-endian, which is this machine's layout.
      size_t count = std::min<size_t>(6, blob->s.size() / 4);
      for (size_t i = 0; i < count; ++i) {
        float f;
        std::memcpy(&f, blob->s.data() + i * 4, 4);
        std::cout << " " << f;
      }
      std::cout << "\n";
    }

    Packer fb;
    fb.map(4);
    fb.str("message_type"); fb.str("feedback");
    fb.str("env_indices");  pack_ndarray(fb, pack_i8(env_idx), "<i8", {batch});
    fb.str("step_ids");     pack_ndarray(fb, pack_i8(step_ids), "<i8", {batch});
    fb.str("data");
    fb.map(5);
    fb.str("obs");        pack_observation(fb, batch);
    {
      std::string rewards(static_cast<size_t>(batch) * 4, '\0');
      for (int i = 0; i < batch; ++i) {
        float r = static_cast<float>(next_float());
        std::memcpy(&rewards[i * 4], &r, 4);
      }
      fb.str("rewards"); pack_ndarray(fb, rewards, "<f4", {batch});
    }
    fb.str("terminated");
    pack_ndarray(fb, std::string(batch, '\0'), "|b1", {batch});
    fb.str("truncated");
    pack_ndarray(fb, std::string(batch, '\0'), "|b1", {batch});
    fb.str("info"); fb.map(0);
    ws.send_binary(fb.data());
  }

  std::cout << "completed " << steps << " infer/action/feedback exchanges\n";

  if (!rtt_ms.empty()) {
    auto stat = [](std::vector<double> v, double q) {
      std::sort(v.begin(), v.end());
      size_t i = static_cast<size_t>(q * (v.size() - 1));
      return v[i];
    };
    auto mean = [](const std::vector<double>& v) {
      return std::accumulate(v.begin(), v.end(), 0.0) / v.size();
    };
    std::cout << std::fixed << std::setprecision(3)
              << "\nboundary cost, " << rtt_ms.size() << " samples"
              << " (batch=" << batch << ", " << g_cameras << " cameras @ "
              << g_img_size << "px, infer payload "
              << infer_bytes / 1024 << " KiB)\n"
              << "  pack     mean " << mean(pack_ms)
              << " ms   p50 " << stat(pack_ms, 0.50)
              << "   p95 " << stat(pack_ms, 0.95) << "\n"
              << "  rtt      mean " << mean(rtt_ms)
              << " ms   p50 " << stat(rtt_ms, 0.50)
              << "   p95 " << stat(rtt_ms, 0.95) << "\n"
              << "  unpack   mean " << mean(unpack_ms)
              << " ms   p50 " << stat(unpack_ms, 0.50)
              << "   p95 " << stat(unpack_ms, 0.95) << "\n"
              << "  TSV\t" << batch << "\t" << g_cameras << "\t" << g_img_size
              << "\t" << infer_bytes << "\t" << mean(pack_ms) << "\t"
              << mean(rtt_ms) << "\t" << mean(unpack_ms) << "\n";
  }
  return 0;
}

}  // namespace

int main(int argc, char** argv) {
  // host port steps batch img_size cameras
  std::string host = argc > 1 ? argv[1] : "127.0.0.1";
  int port = argc > 2 ? std::stoi(argv[2]) : 8123;
  int steps = argc > 3 ? std::stoi(argv[3]) : 20;
  int batch = argc > 4 ? std::stoi(argv[4]) : 1;
  if (argc > 5) g_img_size = std::stoi(argv[5]);
  if (argc > 6) g_cameras = std::stoi(argv[6]);
  try {
    return run(host, port, steps, batch);
  } catch (const std::exception& e) {
    std::cerr << "error: " << e.what() << "\n";
    return 1;
  }
}
