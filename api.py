def synthesize(self, text: str, length_scale: float = 1.0,
               noise_scale: float = 0.667, noise_w_scale: float = 0.8) -> Tuple[bytes, int]:
    request = {
        "text": text,
        "length_scale": length_scale,
        "noise_scale": noise_scale,
        "noise_w": noise_w_scale
    }
    with self.lock:
        try:
            # Envia requisição
            self.process.stdin.write((json.dumps(request) + "\n").encode())
            self.process.stdin.flush()

            # Lê linha JSON byte a byte até '\n' (EVITA pegar áudio)
            line_bytes = b""
            while True:
                ch = self.process.stdout.read(1)
                if not ch:
                    raise RuntimeError("Processo piper morreu antes de enviar JSON")
                if ch == b'\n':
                    break
                line_bytes += ch

            # Decodifica a linha como UTF-8 (o JSON sempre é ASCII/UTF-8)
            line_str = line_bytes.decode('utf-8')
            response = json.loads(line_str)

            num_samples = response.get("num_samples", 0)
            sample_rate = response.get("sample_rate", 22050)

            # Lê exatamente os samples de áudio (16-bit = 2 bytes por amostra)
            raw_audio = self.process.stdout.read(num_samples * 2)
            if len(raw_audio) != num_samples * 2:
                logger.warning(f"[PIPER] Tamanho de áudio inesperado: esperado {num_samples*2}, recebido {len(raw_audio)}")

            return raw_audio, sample_rate

        except (json.JSONDecodeError, KeyError) as e:
            stderr_tail = self._get_stderr_tail()
            logger.error(f"[PIPER] Erro de parsing: {e}. stderr: {stderr_tail[-500:]}")
            self._restart()
            raise RuntimeError(f"Falha na comunicação com piper: {e}")
        except (BrokenPipeError, Exception) as e:
            stderr_tail = self._get_stderr_tail()
            logger.error(f"[PIPER] Exceção: {e}. stderr: {stderr_tail[-500:]}")
            self._restart()
            raise
