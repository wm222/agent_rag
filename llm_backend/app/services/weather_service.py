from typing import Dict, Any, Optional
import aiohttp


class WeatherService:
    GEO_URL = "https://geocoding-api.open-meteo.com/v1/search"
    WEATHER_URL = "https://api.open-meteo.com/v1/forecast"

    WEATHER_CODE_MAP = {
        0: "晴朗",
        1: "大致晴朗",
        2: "局部多云",
        3: "阴天",
        45: "有雾",
        48: "雾凇",
        51: "小毛毛雨",
        53: "中等毛毛雨",
        55: "大毛毛雨",
        61: "小雨",
        63: "中雨",
        65: "大雨",
        71: "小雪",
        73: "中雪",
        75: "大雪",
        80: "小阵雨",
        81: "中等阵雨",
        82: "强阵雨",
        95: "雷暴",
        96: "雷暴并伴有小冰雹",
        99: "雷暴并伴有强冰雹",
    }

    async def geocode(self, location: str) -> Optional[Dict[str, Any]]:
        """
        根据城市名查询经纬度。
        """
        async with aiohttp.ClientSession() as session:
            async with session.get(
                self.GEO_URL,
                params={
                    "name": location,
                    "count": 1,
                    "language": "zh",
                    "format": "json",
                },
                timeout=aiohttp.ClientTimeout(total=20),
            ) as response:
                if response.status != 200:
                    return None

                data = await response.json()

        results = data.get("results") or []
        if not results:
            return None

        item = results[0]

        return {
            "name": item.get("name", location),
            "country": item.get("country", ""),
            "admin1": item.get("admin1", ""),
            "latitude": item.get("latitude"),
            "longitude": item.get("longitude"),
            "timezone": item.get("timezone", "auto"),
        }

    async def get_weather(self, location: str) -> Dict[str, Any]:
        geo = await self.geocode(location)

        if not geo:
            return {
                "status": "error",
                "message": f"没有找到地点：{location}",
            }

        latitude = geo["latitude"]
        longitude = geo["longitude"]

        async with aiohttp.ClientSession() as session:
            async with session.get(
                self.WEATHER_URL,
                params={
                    "latitude": latitude,
                    "longitude": longitude,
                    "current": ",".join([
                        "temperature_2m",
                        "relative_humidity_2m",
                        "apparent_temperature",
                        "precipitation",
                        "weather_code",
                        "wind_speed_10m",
                        "wind_direction_10m",
                    ]),
                    "daily": ",".join([
                        "weather_code",
                        "temperature_2m_max",
                        "temperature_2m_min",
                        "precipitation_probability_max",
                    ]),
                    "forecast_days": 3,
                    "timezone": "auto",
                },
                timeout=aiohttp.ClientTimeout(total=20),
            ) as response:
                if response.status != 200:
                    detail = await response.text()
                    return {
                        "status": "error",
                        "message": f"天气接口调用失败：{response.status}, {detail}",
                    }

                data = await response.json()

        current = data.get("current", {})
        daily = data.get("daily", {})

        code = current.get("weather_code")
        weather_text = self.WEATHER_CODE_MAP.get(code, f"未知天气代码 {code}")

        place_name = geo.get("name", location)
        admin1 = geo.get("admin1", "")
        country = geo.get("country", "")

        display_location = "，".join([x for x in [country, admin1, place_name] if x])

        answer_lines = [
            f"{display_location} 当前天气：",
            f"- 天气：{weather_text}",
            f"- 当前气温：{current.get('temperature_2m')}℃",
            f"- 体感温度：{current.get('apparent_temperature')}℃",
            f"- 相对湿度：{current.get('relative_humidity_2m')}%",
            f"- 降水量：{current.get('precipitation')} mm",
            f"- 风速：{current.get('wind_speed_10m')} km/h",
        ]

        dates = daily.get("time", [])
        max_temps = daily.get("temperature_2m_max", [])
        min_temps = daily.get("temperature_2m_min", [])
        rain_probs = daily.get("precipitation_probability_max", [])
        daily_codes = daily.get("weather_code", [])

        if dates:
            answer_lines.append("")
            answer_lines.append("未来 3 天预报：")

            for i, date in enumerate(dates):
                day_code = daily_codes[i] if i < len(daily_codes) else None
                day_weather = self.WEATHER_CODE_MAP.get(day_code, f"未知天气代码 {day_code}")

                max_temp = max_temps[i] if i < len(max_temps) else "未知"
                min_temp = min_temps[i] if i < len(min_temps) else "未知"
                rain_prob = rain_probs[i] if i < len(rain_probs) else "未知"

                answer_lines.append(
                    f"- {date}：{day_weather}，{min_temp}℃ ~ {max_temp}℃，降水概率 {rain_prob}%"
                )

        return {
            "status": "success",
            "location": display_location,
            "latitude": latitude,
            "longitude": longitude,
            "current": current,
            "daily": daily,
            "answer": "\n".join(answer_lines),
        }