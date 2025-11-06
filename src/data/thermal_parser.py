# Port of scripts/code.c to Python, preserving structure for side-by-side comparison
# Structs -> classes; arrays -> lists with explicit type annotations

from math import pow, sqrt

# Defines / constants
EMISSIVITY_COEFFICIENT = 0.96
Tr_Shift = 8

def SUBPAGE_NUMBER(x: int) -> int:
	return ((x % 2) ^ ((x >> 5) & 0x01))

MLX_EEPROM_BASE_ADDR = 0x2400
MLX_CONTROL_REG_ADDR = 0x800D
MLX_STATUS_REG_ADDR = 0x8000

MLX90640_RES_COLS = 32
MLX90640_RES_ROWS = 24

def REV_BYTES(x: int) -> int:
	return (((x) & 0x00FF) << 8) | (((x) & 0xFF00) >> 8)



class mlx90640_parameters:
	def __init__(self) -> None:
		# Scalars
		self.kVdd: int = 0
		self.Vdd25: int = 0
		self.KvPTAT: float = 0.0
		self.KtPTAT: float = 0.0
		self.vPTAT25: int = 0
		self.alphaPTAT: float = 0.0
		self.gainEE: int = 0
		self.KsTa: float = 0.0
		self.ct: list[int] = [0, 0, 0, 0]
		self.ksTo: list[float] = [0.0, 0.0, 0.0, 0.0]
		self.alphaCorr: list[float] = [0.0, 0.0, 0.0, 0.0]
		self.cpAlpha: list[float] = [0.0, 0.0]
		self.cpOffset: list[int] = [0, 0]
		self.CpKv: float = 0.0
		self.CpKta: float = 0.0
		self.Tgc: float = 0.0
		self.resolutionEE: int = 0  # uint8_t in C
		self.currentRefreshRate: int = 0  # uint8_t in C (keep even if unused)

		# Arrays for each pixel (24x32 = 768)
		self.offset: list[int] = [0] * 768
		self.ilChessC: list[float] = [0.0] * 3
		self.alpha: list[float] = [0.0] * 768
		self.kv: list[float] = [0.0] * 768
		self.kta: list[float] = [0.0] * 768


class mlx_frame_parameters:
	def __init__(self) -> None:
		self.resolution_reg: int = 0  # uint8_t in C
		self.vdd_pix: float = 0.0
		self.ta_ptat: int = 0
		self.ta_vbe: int = 0
		self.gain: int = 0
		self.cp_sp: int = 0


def mlx_parse_eeprom_parameters(params: mlx90640_parameters, mlx90640_eeprom: list[int]) -> None:
	params.kVdd = (mlx90640_eeprom[0x33] & 0xFF00) >> 8
	params.Vdd25 = (mlx90640_eeprom[0x33] & 0x00FF)
	params.kVdd = (params.kVdd - 256) if (params.kVdd > 127) else params.kVdd
	params.kVdd = (params.kVdd * (1 << 5))
	params.Vdd25 = (params.Vdd25 - 256) << 5
	params.Vdd25 -= (1 << 13)
	params.KtPTAT = (mlx90640_eeprom[0x32] & 0x03FF)
	params.KtPTAT = (params.KtPTAT - 1024) if (params.KtPTAT > 511) else params.KtPTAT
	params.KtPTAT = params.KtPTAT / (1 << 3)
	params.KvPTAT = (mlx90640_eeprom[0x32] & 0xFC00) >> 10
	params.KvPTAT = (params.KvPTAT - 64) if (params.KvPTAT > 31) else params.KvPTAT
	params.KvPTAT = (params.KvPTAT / (1 << 12))
	params.vPTAT25 = mlx90640_eeprom[0x31]
	params.vPTAT25 = (params.vPTAT25 - 65536) if (params.vPTAT25 > 32767) else params.vPTAT25
	params.alphaPTAT = ((mlx90640_eeprom[0x10] & 0xF000) >> 12)
	params.alphaPTAT = 8.0 + (params.alphaPTAT / 4)

	# Offset calculations
	offset_avg = mlx90640_eeprom[0x11]
	offset_avg = (offset_avg - 65536) if (offset_avg > 32767) else offset_avg
	row_scale = (mlx90640_eeprom[0x10] & 0x0F00) >> 8
	column_scale = (mlx90640_eeprom[0x10] & 0x00F0) >> 4
	remnant_scale = (mlx90640_eeprom[0x10] & 0x000F)
	row_occ: list[int] = [0] * 24
	column_occ: list[int] = [0] * 32
	for i in range(6):
		row_occ[i * 4] = mlx90640_eeprom[0x12 + i] & 0x000F
		row_occ[i * 4] = (row_occ[i * 4] - 16) if (row_occ[i * 4] > 7) else row_occ[i * 4]
		row_occ[i * 4 + 1] = (mlx90640_eeprom[0x12 + i] & 0x00F0) >> 4
		row_occ[i * 4 + 1] = (row_occ[i * 4 + 1] - 16) if (row_occ[i * 4 + 1] > 7) else row_occ[i * 4 + 1]
		row_occ[i * 4 + 2] = (mlx90640_eeprom[0x12 + i] & 0x0F00) >> 8
		row_occ[i * 4 + 2] = (row_occ[i * 4 + 2] - 16) if (row_occ[i * 4 + 2] > 7) else row_occ[i * 4 + 2]
		row_occ[i * 4 + 3] = (mlx90640_eeprom[0x12 + i] & 0xF000) >> 12
		row_occ[i * 4 + 3] = (row_occ[i * 4 + 3] - 16) if (row_occ[i * 4 + 3] > 7) else row_occ[i * 4 + 3]

	for i in range(8):
		column_occ[i * 4] = mlx90640_eeprom[0x18 + i] & 0x000F
		column_occ[i * 4] = (column_occ[i * 4] - 16) if (column_occ[i * 4] > 7) else column_occ[i * 4]
		column_occ[i * 4 + 1] = (mlx90640_eeprom[0x18 + i] & 0x00F0) >> 4
		column_occ[i * 4 + 1] = (column_occ[i * 4 + 1] - 16) if (column_occ[i * 4 + 1] > 7) else column_occ[i * 4 + 1]
		column_occ[i * 4 + 2] = (mlx90640_eeprom[0x18 + i] & 0x0F00) >> 8
		column_occ[i * 4 + 2] = (column_occ[i * 4 + 2] - 16) if (column_occ[i * 4 + 2] > 7) else column_occ[i * 4 + 2]
		column_occ[i * 4 + 3] = (mlx90640_eeprom[0x18 + i] & 0xF000) >> 12
		column_occ[i * 4 + 3] = (column_occ[i * 4 + 3] - 16) if (column_occ[i * 4 + 3] > 7) else column_occ[i * 4 + 3]

	for i in range(24):
		for j in range(32):
			idx = 32 * i + j
			params.offset[idx] = (mlx90640_eeprom[(idx) + 0x40] & 0xFC00) >> 10
			params.offset[idx] = (params.offset[idx] - 64) if (params.offset[idx] > 31) else params.offset[idx]
			params.offset[idx] = (params.offset[idx] * (1 << remnant_scale))
			params.offset[idx] += offset_avg + (row_occ[i] * (1 << row_scale)) + (column_occ[j] * (1 << column_scale))

	params.ilChessC[0] = mlx90640_eeprom[0x35] & 0x003F
	params.ilChessC[0] = (params.ilChessC[0] - 64) if (params.ilChessC[0] > 31) else params.ilChessC[0]
	params.ilChessC[0] = params.ilChessC[0] / (1 << 4)
	params.ilChessC[1] = (mlx90640_eeprom[0x35] & 0x07C0) >> 6
	params.ilChessC[1] = (params.ilChessC[1] - 32) if (params.ilChessC[1] > 15) else params.ilChessC[1]
	params.ilChessC[1] = params.ilChessC[1] / (1 << 1)
	params.ilChessC[2] = (mlx90640_eeprom[0x35] & 0xF800) >> 11
	params.ilChessC[2] = (params.ilChessC[2] - 32) if (params.ilChessC[2] > 15) else params.ilChessC[2]
	params.ilChessC[2] = params.ilChessC[2] / (1 << 3)

	alpha_avg = mlx90640_eeprom[0x21]
	alpha_scale = 30 + ((mlx90640_eeprom[0x20] & 0xF000) >> 12)
	alpha_row_scale = (mlx90640_eeprom[0x20] & 0x0F00) >> 8
	alpha_column_scale = (mlx90640_eeprom[0x20] & 0x00F0) >> 4
	alpha_remnant_scale = (mlx90640_eeprom[0x20] & 0x000F)
	row_acc: list[int] = [0] * 24
	column_acc: list[int] = [0] * 32

	for i in range(6):
		row_acc[i * 4] = mlx90640_eeprom[0x22 + i] & 0x000F
		row_acc[i * 4] = (row_acc[i * 4] - 16) if (row_acc[i * 4] > 7) else row_acc[i * 4]
		row_acc[i * 4 + 1] = (mlx90640_eeprom[0x22 + i] & 0x00F0) >> 4
		row_acc[i * 4 + 1] = (row_acc[i * 4 + 1] - 16) if (row_acc[i * 4 + 1] > 7) else row_acc[i * 4 + 1]
		row_acc[i * 4 + 2] = (mlx90640_eeprom[0x22 + i] & 0x0F00) >> 8
		row_acc[i * 4 + 2] = (row_acc[i * 4 + 2] - 16) if (row_acc[i * 4 + 2] > 7) else row_acc[i * 4 + 2]
		row_acc[i * 4 + 3] = (mlx90640_eeprom[0x22 + i] & 0xF000) >> 12
		row_acc[i * 4 + 3] = (row_acc[i * 4 + 3] - 16) if (row_acc[i * 4 + 3] > 7) else row_acc[i * 4 + 3]

	for i in range(8):
		column_acc[i * 4] = mlx90640_eeprom[0x28 + i] & 0x000F
		column_acc[i * 4] = (column_acc[i * 4] - 16) if (column_acc[i * 4] > 7) else column_acc[i * 4]
		column_acc[i * 4 + 1] = (mlx90640_eeprom[0x28 + i] & 0x00F0) >> 4
		column_acc[i * 4 + 1] = (column_acc[i * 4 + 1] - 16) if (column_acc[i * 4 + 1] > 7) else column_acc[i * 4 + 1]
		column_acc[i * 4 + 2] = (mlx90640_eeprom[0x28 + i] & 0x0F00) >> 8
		column_acc[i * 4 + 2] = (column_acc[i * 4 + 2] - 16) if (column_acc[i * 4 + 2] > 7) else column_acc[i * 4 + 2]
		column_acc[i * 4 + 3] = (mlx90640_eeprom[0x28 + i] & 0xF000) >> 12
		column_acc[i * 4 + 3] = (column_acc[i * 4 + 3] - 16) if (column_acc[i * 4 + 3] > 7) else column_acc[i * 4 + 3]

	for i in range(24):
		for j in range(32):
			idx = 32 * i + j
			params.alpha[idx] = ((mlx90640_eeprom[(idx) + 0x40] & 0x03F0) >> 4)
			params.alpha[idx] = (params.alpha[idx] - 64) if (params.alpha[idx] > 31) else params.alpha[idx]
			params.alpha[idx] = params.alpha[idx] * (1 << alpha_remnant_scale)
			params.alpha[idx] = params.alpha[idx] + alpha_avg + (row_acc[i] * (1 << alpha_row_scale)) + (column_acc[j] * (1 << alpha_column_scale))
			params.alpha[idx] = params.alpha[idx] / pow(2, alpha_scale)

	Kvscale = (mlx90640_eeprom[0x38] & 0x0F00) >> 8
	KvOEArray: list[int] = [0, 0, 0, 0]  # Row-Column; Odd - 0; Even - 1
	KvOEArray[0] = (mlx90640_eeprom[0x34] & 0xF000) >> 12  # Odd Odd
	KvOEArray[0] = (KvOEArray[0] - 16) if (KvOEArray[0] > 7) else KvOEArray[0]
	KvOEArray[1] = (mlx90640_eeprom[0x34] & 0x00F0) >> 4   # Odd Even
	KvOEArray[1] = (KvOEArray[1] - 16) if (KvOEArray[1] > 7) else KvOEArray[1]
	KvOEArray[2] = (mlx90640_eeprom[0x34] & 0x0F00) >> 8   # Even Odd
	KvOEArray[2] = (KvOEArray[2] - 16) if (KvOEArray[2] > 7) else KvOEArray[2]
	KvOEArray[3] = (mlx90640_eeprom[0x34] & 0x000F)        # Even Even
	KvOEArray[3] = (KvOEArray[3] - 16) if (KvOEArray[3] > 7) else KvOEArray[3]
	for i in range(24):
		for j in range(32):
			idx = 32 * i + j
			params.kv[idx] = (KvOEArray[(i % 2) * 2 + (j % 2)])
			params.kv[idx] = (params.kv[idx] / (1 << Kvscale))

	KtaOEArrau: list[int] = [0, 0, 0, 0]
	KtaOEArrau[0] = (mlx90640_eeprom[0x36] & 0xFF00) >> 8  # Odd Odd
	KtaOEArrau[0] = (KtaOEArrau[0] - 256) if (KtaOEArrau[0] > 127) else KtaOEArrau[0]
	KtaOEArrau[1] = (mlx90640_eeprom[0x37] & 0xFF00) >> 8  # Odd Even
	KtaOEArrau[1] = (KtaOEArrau[1] - 256) if (KtaOEArrau[1] > 127) else KtaOEArrau[1]
	KtaOEArrau[2] = (mlx90640_eeprom[0x36] & 0x00FF)       # Even Odd
	KtaOEArrau[2] = (KtaOEArrau[2] - 256) if (KtaOEArrau[2] > 127) else KtaOEArrau[2]
	KtaOEArrau[3] = (mlx90640_eeprom[0x37] & 0x00FF)       # Even Even
	KtaOEArrau[3] = (KtaOEArrau[3] - 256) if (KtaOEArrau[3] > 127) else KtaOEArrau[3]
	KtaScale1 = 8 + ((mlx90640_eeprom[0x38] & 0x00F0) >> 4)
	KtaScale2 = (mlx90640_eeprom[0x38] & 0x000F)
	for i in range(24):
		for j in range(32):
			idx = 32 * i + j
			ktaEE = ((mlx90640_eeprom[0x40 + 32 * i + j] & 0x000E) // 2)
			ktaEE = (ktaEE - 8) if (ktaEE > 3) else ktaEE
			params.kta[idx] = (KtaOEArrau[(i % 2) * 2 + (j % 2)] + (ktaEE * (1 << KtaScale2)))
			params.kta[idx] = (params.kta[idx] / (1 << KtaScale1))

	params.gainEE = mlx90640_eeprom[0x30]
	params.gainEE = (params.gainEE - 65536) if (params.gainEE > 32767) else params.gainEE

	params.KsTa = (mlx90640_eeprom[0x3C] & 0xFF00) >> 8
	params.KsTa = (params.KsTa - 256) if (params.KsTa > 127) else params.KsTa
	params.KsTa = (params.KsTa / (1 << 13))

	step = 10 * ((mlx90640_eeprom[0x3F] & 0x3000) >> 12)
	params.ct[0] = -40
	params.ct[1] = 0
	params.ct[2] = ((mlx90640_eeprom[0x3F] & 0x00F0) >> 4) * step
	params.ct[3] = ((mlx90640_eeprom[0x3F] & 0x0F00) >> 8) * step + params.ct[2]

	KsToScale = 8 + (mlx90640_eeprom[0x3F] & 0x000F)
	params.ksTo[0] = (mlx90640_eeprom[0x3D] & 0x00FF)
	params.ksTo[0] = (params.ksTo[0] - 256) if (params.ksTo[0] > 127) else params.ksTo[0]
	params.ksTo[0] = params.ksTo[0] / (1 << KsToScale)
	params.ksTo[1] = (mlx90640_eeprom[0x3D] & 0xFF00) >> 8
	params.ksTo[1] = (params.ksTo[1] - 256) if (params.ksTo[1] > 127) else params.ksTo[1]
	params.ksTo[1] = params.ksTo[1] / (1 << KsToScale)
	params.ksTo[2] = (mlx90640_eeprom[0x3E] & 0x00FF)
	params.ksTo[2] = (params.ksTo[2] - 256) if (params.ksTo[2] > 127) else params.ksTo[2]
	params.ksTo[2] = params.ksTo[2] / (1 << KsToScale)
	params.ksTo[3] = (mlx90640_eeprom[0x3E] & 0xFF00) >> 8
	params.ksTo[3] = (params.ksTo[3] - 256) if (params.ksTo[3] > 127) else params.ksTo[3]
	params.ksTo[3] = params.ksTo[3] / (1 << KsToScale)

	params.alphaCorr[0] = 1 / (1 + params.ksTo[0] * 40)
	params.alphaCorr[1] = 1
	params.alphaCorr[2] = 1 + params.ksTo[2] * params.ct[2]
	params.alphaCorr[3] = (1 + params.ksTo[3] * (params.ct[3] - params.ct[2])) * (params.alphaCorr[2])

	alphaCPscale = 27 + ((mlx90640_eeprom[0x20] & 0xF000) >> 12)
	CP_P1_P0_ratio = ((mlx90640_eeprom[0x39] & 0xFC00) >> 10)
	CP_P1_P0_ratio = (CP_P1_P0_ratio - 64) if (CP_P1_P0_ratio > 31) else CP_P1_P0_ratio
	params.cpAlpha[0] = (mlx90640_eeprom[0x39] & 0x03FF) / (1 << alphaCPscale)
	params.cpAlpha[1] = (params.cpAlpha[0] * (1 + CP_P1_P0_ratio / (1 << 7)))

	params.cpOffset[0] = (mlx90640_eeprom[0x3A] & 0x00FF)
	params.cpOffset[0] = (params.cpOffset[0] - 1024) if (params.cpOffset[0] > 511) else params.cpOffset[0]
	params.cpOffset[1] = (mlx90640_eeprom[0x3A] & 0xFC00) >> 10
	params.cpOffset[1] = (params.cpOffset[1] - 64) if (params.cpOffset[1] > 31) else params.cpOffset[1]
	params.cpOffset[1] += params.cpOffset[0]

	params.CpKv = (mlx90640_eeprom[0x3B] & 0xFF00) >> 8
	params.CpKv = (params.CpKv - 256) if (params.CpKv > 127) else params.CpKv
	params.CpKv = (params.CpKv / (1 << Kvscale))

	params.CpKta = (mlx90640_eeprom[0x3B] & 0x00FF)
	params.CpKta = (params.CpKta - 256) if (params.CpKta > 127) else params.CpKta
	params.CpKta = (params.CpKta / (1 << KtaScale1))

	params.Tgc = (mlx90640_eeprom[0x3C] & 0x00FF)
	params.Tgc = (params.Tgc - 256) if (params.Tgc > 127) else params.Tgc
	params.Tgc = (params.Tgc / (1 << 5))

	params.resolutionEE = (mlx90640_eeprom[0x38] & 0x3000) >> 12
 


def get_temperature_of_pixel(params: mlx90640_parameters, frame_params: mlx_frame_parameters, frame_data: list[int], pixel_index: int) -> float:
	resolution_corr = 0.0
	resolution_corr = (1 << params.resolutionEE) / (1 << frame_params.resolution_reg)
	vdd = 3.3 + (frame_params.vdd_pix * resolution_corr - params.Vdd25) / params.kVdd
	deltaV = (frame_params.vdd_pix - params.Vdd25) / params.kVdd
	vPTATart = (1 << 18) * (frame_params.ta_ptat / (frame_params.ta_vbe + params.alphaPTAT * frame_params.ta_ptat))
	Ta = 25 + (((vPTATart / (1 + deltaV * params.KvPTAT)) - params.vPTAT25) / params.KtPTAT)

	kGain = (float(params.gainEE)) / frame_params.gain
	pix_gain = frame_data[pixel_index]
	pix_gain = (pix_gain - 65536) if (pix_gain > 32767) else pix_gain
	pix_gain = pix_gain * kGain
	pix_os = pix_gain - (params.offset[pixel_index]) * (1 + params.kta[pixel_index] * (Ta - 25)) * (1 + params.kv[pixel_index] * (vdd - 3.3))

	pix_os = pix_os / EMISSIVITY_COEFFICIENT

	sub_page_gain = frame_params.cp_sp
	sub_page_gain = sub_page_gain * kGain
	sub_page_offset = sub_page_gain - params.cpOffset[SUBPAGE_NUMBER(pixel_index)] * (1 + params.CpKta * (Ta - 25)) * (1 + params.CpKv * (vdd - 3.3))
	Vir_compensated = pix_os - params.Tgc * (sub_page_offset)
	alpha_compensated = params.alpha[pixel_index] - params.Tgc * params.cpAlpha[SUBPAGE_NUMBER(pixel_index)]
	alpha_compensated = alpha_compensated * (1 + params.KsTa * (Ta - 25))
	TaK4 = (Ta + 273.15) * (Ta + 273.15) * (Ta + 273.15) * (Ta + 273.15)
	TrK4 = (Ta + 273.15 - Tr_Shift) * (Ta + 273.15 - Tr_Shift) * (Ta + 273.15 - Tr_Shift) * (Ta + 273.15 - Tr_Shift)
	Ta_r = TrK4 - (TrK4 - TaK4) / EMISSIVITY_COEFFICIENT
	Sx = alpha_compensated * alpha_compensated * alpha_compensated * Vir_compensated + alpha_compensated * alpha_compensated * alpha_compensated * alpha_compensated * Ta_r
	Sx = sqrt(sqrt(Sx)) * params.ksTo[1]
	To = sqrt(sqrt(Ta_r + (Vir_compensated / (Sx + alpha_compensated * (1 - 273.15 * params.ksTo[1]))))) - 273.15

	range_idx = 0
	if To < params.ct[1]:
		range_idx = 0
	if To >= params.ct[1] and To < params.ct[2]:
		range_idx = 1
	if To >= params.ct[2] and To < params.ct[3]:
		range_idx = 2
	if To >= params.ct[3]:
		range_idx = 3
	To = sqrt(sqrt(Ta_r + (Vir_compensated / (params.alphaCorr[range_idx] * alpha_compensated * (1 + params.ksTo[range_idx] * (To - params.ct[range_idx])))))) - 273.15
	return To


def mlx_parse_frame_parameters(subpage_0_frame: list[int], subpage_1_frame: list[int], params: mlx90640_parameters, subpage_0_params: mlx_frame_parameters, subpage_1_params: mlx_frame_parameters, control_register_value: int) -> None:
	# Now comes the part I don't like!
	# Each pixel has its own subpage, and right now we support only chess mode.
	# So we have to read temperature data from subpage-0 and subpage-1 and depending on those values, we write to the temperature data array.

	subpage_0_params.resolution_reg = (control_register_value & 0x0C00) >> 10
	subpage_1_params.resolution_reg = subpage_0_params.resolution_reg
	subpage_0_params.vdd_pix = (subpage_0_frame[0x072A - 0x0400])
	subpage_1_params.vdd_pix = (subpage_1_frame[0x072A - 0x0400])
	subpage_0_params.vdd_pix = (subpage_0_params.vdd_pix - 65536) if (subpage_0_params.vdd_pix > 32767) else subpage_0_params.vdd_pix
	subpage_1_params.vdd_pix = (subpage_1_params.vdd_pix - 65536) if (subpage_1_params.vdd_pix > 32767) else subpage_1_params.vdd_pix
	subpage_0_params.ta_ptat = (subpage_0_frame[0x0720 - 0x0400])
	subpage_1_params.ta_ptat = (subpage_1_frame[0x0720 - 0x0400])
	subpage_0_params.ta_ptat = (subpage_0_params.ta_ptat - 65536) if (subpage_0_params.ta_ptat > 32767) else subpage_0_params.ta_ptat
	subpage_1_params.ta_ptat = (subpage_1_params.ta_ptat - 65536) if (subpage_1_params.ta_ptat > 32767) else subpage_1_params.ta_ptat
	subpage_0_params.ta_vbe = (subpage_0_frame[0x0700 - 0x0400])
	subpage_1_params.ta_vbe = (subpage_1_frame[0x0700 - 0x0400])
	subpage_0_params.ta_vbe = (subpage_0_params.ta_vbe - 65536) if (subpage_0_params.ta_vbe > 32767) else subpage_0_params.ta_vbe
	subpage_1_params.ta_vbe = (subpage_1_params.ta_vbe - 65536) if (subpage_1_params.ta_vbe > 32767) else subpage_1_params.ta_vbe
	subpage_0_params.gain = (subpage_0_frame[0x070A - 0x0400])
	subpage_1_params.gain = (subpage_1_frame[0x070A - 0x0400])
	subpage_0_params.gain = (subpage_0_params.gain - 65536) if (subpage_0_params.gain > 32767) else subpage_0_params.gain
	subpage_1_params.gain = (subpage_1_params.gain - 65536) if (subpage_1_params.gain > 32767) else subpage_1_params.gain
	subpage_0_params.cp_sp = (subpage_0_frame[0x0708 - 0x0400])
	subpage_1_params.cp_sp = (subpage_1_frame[0x0728 - 0x0400])
	subpage_0_params.cp_sp = (subpage_0_params.cp_sp - 65536) if (subpage_0_params.cp_sp > 32767) else subpage_0_params.cp_sp
	subpage_1_params.cp_sp = (subpage_1_params.cp_sp - 65536) if (subpage_1_params.cp_sp > 32767) else subpage_1_params.cp_sp

def get_pixel_in_image_mode(params: mlx90640_parameters, frame_params: mlx_frame_parameters, frame_data: list[int], pixel_index: int) -> float:

    resolution_corr = (1 << params.resolutionEE) / (1 << frame_params.resolution_reg)
    vdd = 3.3 + (frame_params.vdd_pix * resolution_corr - params.Vdd25) / params.kVdd
    deltaV = (frame_params.vdd_pix - params.Vdd25) / params.kVdd
    vPTATart = (1 << 18) * (frame_params.ta_ptat / (frame_params.ta_vbe + params.alphaPTAT * frame_params.ta_ptat))
    Ta = 25 + (((vPTATart / (1 + deltaV * params.KvPTAT)) - params.vPTAT25) / params.KtPTAT)

    # Gain Compensation
    kGain = (float(params.gainEE)) / frame_params.gain
    pix_gain = (frame_data[pixel_index] - 65536) if (frame_data[pixel_index] > 32767) else frame_data[pixel_index]

    pix_gain = pix_gain * kGain
    
    pix_os = pix_gain - (params.offset[pixel_index]) * (1 + params.kta[pixel_index] * (Ta - 25)) * (1 + params.kv[pixel_index] * (vdd - 3.3))
    sub_page_gain = frame_params.cp_sp
    sub_page_gain = sub_page_gain * kGain
    sub_page_offset = sub_page_gain - params.cpOffset[SUBPAGE_NUMBER(pixel_index)] * (1 + params.CpKta * (Ta - 25)) * (1 + params.CpKv * (vdd - 3.3))
    pix_os = pix_os - params.Tgc * (sub_page_offset)
    alpha_compensated = params.alpha[pixel_index] - params.Tgc * params.cpAlpha[SUBPAGE_NUMBER(pixel_index)]
    alpha_compensated = alpha_compensated * (1 + params.KsTa * (Ta - 25))

    return pix_os * alpha_compensated