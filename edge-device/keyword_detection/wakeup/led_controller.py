from abc import ABC, abstractmethod
from collections import namedtuple
import numpy as np
from spidev import SpiDev
import time
import math

# 색상을 표현하기 위한 namedtuple (r, g, b)
Color = namedtuple("Color", ["r", "g", "b"])

class Strip:
    """
    WS2812 LED 스트립 제어 클래스
    """
    def __init__(self, backend: "WS2812StripDriver"):
        self._led_count = backend.get_led_count()
        self._brightness = 1.0
        self._pixels: list[Color] = [Color(0, 0, 0)] * self._led_count
        self._backend = backend

    def set_pixel_color(self, i: int, color: Color) -> None:
        """
        특정 인덱스 LED 색상을 설정 (show() 호출 전까지 버퍼에 저장)
        """
        self._pixels[i] = color

    def show(self) -> None:
        """
        현재 버퍼에 저장된 색상을 LED 스트립에 출력
        """
        buffer = np.array(
            [
                np.array([pixel.g * self._brightness, pixel.r * self._brightness, pixel.b * self._brightness])
                for pixel in self._pixels
            ],
            dtype=np.uint8,
        )
        self._backend.write(buffer)

    def clear(self) -> None:
        """
        LED 스트립의 모든 LED를 off로 설정하고, 버퍼를 초기화
        """
        self._pixels = [Color(0, 0, 0)] * self._led_count
        self._backend.clear()

    def set_brightness(self, brightness: float) -> None:
        """
        LED 밝기를 0.0 ~ 1.0 사이 값으로 설정
        """
        self._brightness = max(min(brightness, 1.0), 0.0)

    def num_pixels(self) -> int:
        """
        LED 개수 반환
        """
        return self._led_count

    def get_brightness(self) -> float:
        """
        현재 밝기 반환
        """
        return self._brightness

    def set_all_pixels(self, color: Color) -> None:
        """
        모든 LED의 색상을 동일하게 설정 (show() 호출 전까지 적용되지 않음)
        """
        self._pixels = [color] * self._led_count

class WS2812StripDriver(ABC):
    """
    WS2812 LED 스트립 드라이버를 위한 추상 베이스 클래스
    """
    @abstractmethod
    def write(self, colors: np.ndarray) -> None:
        pass

    @abstractmethod
    def clear(self) -> None:
        pass

    @abstractmethod
    def get_led_count(self) -> int:
        pass

    def get_strip(self) -> Strip:
        return Strip(self)

class WS2812SpiDriver(WS2812StripDriver):
    """
    Raspberry Pi의 SPI 인터페이스를 이용하여 WS2812 LED 스트립을 제어하는 드라이버
    """
    LED_ZERO: int = 0b11000000  # 0 비트 패턴 (LED off)
    LED_ONE: int = 0b11111100   # 1 비트 패턴 (LED on)
    PREAMBLE: int = 42          # 리셋 신호를 위한 프리앰블 길이 (42 바이트)

    def __init__(self, spi_bus: int, spi_device: int, led_count: int):
        self._led_count = led_count
        self._device = SpiDev()
        self._device.open(spi_bus, spi_device)
        self._device.max_speed_hz = 6_500_000
        self._device.mode = 0b00
        self._device.lsbfirst = False

        # LED 전체를 off 상태로 만드는 clear 버퍼 생성
        self._clear_buffer = np.zeros(WS2812SpiDriver.PREAMBLE + led_count * 24, dtype=np.uint8)
        self._clear_buffer[WS2812SpiDriver.PREAMBLE:] = np.full(
            led_count * 24, WS2812SpiDriver.LED_ZERO, dtype=np.uint8
        )

        # 전송할 데이터 버퍼 초기화
        self._buffer = np.zeros(WS2812SpiDriver.PREAMBLE + led_count * 24, dtype=np.uint8)

    def write(self, buffer: np.ndarray) -> None:
        """
        GRB 순서의 색상 배열(buffer)을 받아서 SPI로 전송.
        numpy의 unpackbits()를 사용하여 각 색상의 비트를 0 또는 1 배열로 변환한 후,
        LED_ONE과 LED_ZERO에 매핑하여 SPI 데이터 버퍼에 저장합니다.
        """
        flattened_colors = buffer.ravel()
        color_bits = np.unpackbits(flattened_colors)
        self._buffer[WS2812SpiDriver.PREAMBLE:] = np.where(
            color_bits == 1, WS2812SpiDriver.LED_ONE, WS2812SpiDriver.LED_ZERO
        )
        self._device.writebytes2(self._buffer)

    def clear(self) -> None:
        """
        LED 스트립의 모든 LED를 off로 전환하는 clear 명령어
        """
        self._device.writebytes2(self._clear_buffer)

    def get_led_count(self) -> int:
        return self._led_count

# LED 효과 함수들
def all_white_on(strip: Strip) -> None:
    """
    전체 LED를 흰색으로 켭니다.
    """
    strip.set_all_pixels(Color(255, 255, 255))
    strip.show()

def all_off(strip: Strip) -> None:
    """
    전체 LED를 OFF 합니다.
    """
    strip.clear()

def white_rotate(strip: Strip, iterations: int = None, delay: float = 0.1) -> None:
    """
    흰색 회전 효과를 구현합니다.
    :param strip: LED 스트립 객체
    :param iterations: 회전 반복 횟수 (기본값은 LED 개수의 5배)
    :param delay: 각 회전 사이의 지연 시간 (초)
    """
    led_count = strip.num_pixels()
    if iterations is None:
        iterations = led_count * 5
    # 패턴: 하나의 LED는 흰색, 나머지는 꺼짐 상태
    pattern = [Color(255, 255, 255)] + [Color(0, 0, 0)] * (led_count - 1)
    for _ in range(iterations):
        first = pattern.pop(0)
        pattern.append(first)
        for i in range(led_count):
            strip.set_pixel_color(i, pattern[i])
        strip.show()
        time.sleep(delay)

def all_blue_on(strip: Strip) -> None:
    """
    전체 LED를 파란색으로 켭니다.
    """
    strip.set_all_pixels(Color(0, 0, 255))
    strip.show()

def all_green_on(strip: Strip) -> None:
    """
    전체 LED를 초록색으로 켭니다.
    """
    strip.set_all_pixels(Color(0, 255, 0))
    strip.show()

def white_breathing(strip: Strip, cycles: int = 5, steps: int = 50, delay: float = 0.05) -> None:
    """
    흰색 Breathing 효과를 구현합니다.
    :param strip: LED 스트립 객체
    :param cycles: Breathing 사이클 횟수
    :param steps: 한 사이클 내 밝기 변화 단계 수 (증가 + 감소)
    :param delay: 각 단계 사이의 지연 시간 (초)
    """
    original_brightness = strip.get_brightness()
    for _ in range(cycles):
        # 밝기 증가 (0 ~ 1)
        for step in range(steps):
            brightness = step / (steps - 1)
            strip.set_brightness(brightness)
            all_white_on(strip)
            time.sleep(delay)
        # 밝기 감소 (1 ~ 0)
        for step in range(steps):
            brightness = 1 - (step / (steps - 1))
            strip.set_brightness(brightness)
            all_white_on(strip)
            time.sleep(delay)
    # 밝기 원복
    strip.set_brightness(original_brightness)
    all_white_on(strip)

def main():
    SPI_BUS = 0
    SPI_DEVICE = 0
    LED_COUNT = 24

    # SPI 드라이버와 Strip 객체 초기화
    driver = WS2812SpiDriver(SPI_BUS, SPI_DEVICE, LED_COUNT)
    strip = driver.get_strip()

    try:
        print("1. 전체 흰색 ON")
        all_white_on(strip)
        time.sleep(2)

        print("2. 전체 OFF")
        all_off(strip)
        time.sleep(2)

        print("3. 흰색 회전 효과")
        white_rotate(strip, delay=0.1)
        time.sleep(2)

        print("4. 파란색 전체 ON")
        all_blue_on(strip)
        time.sleep(2)

        print("5. 초록색 전체 ON")
        all_green_on(strip)
        time.sleep(2)

        print("6. 흰색 Breathing 효과")
        white_breathing(strip, cycles=3, steps=50, delay=0.05)
        time.sleep(2)

        print("모든 효과 종료, LED OFF")
        all_off(strip)
    finally:
        # SPI 자원 해제
        driver._device.close()

if __name__ == '__main__':
    main()
