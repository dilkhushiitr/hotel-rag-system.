import 'package:flutter_riverpod/flutter_riverpod.dart';
import 'package:moneylens/core/constants/app_constants.dart';
import 'package:moneylens/database/database_helper.dart';
import 'package:moneylens/models/transaction_model.dart';
import 'package:moneylens/repositories/category_repository.dart';
import 'package:moneylens/repositories/merchant_repository.dart';
import 'package:moneylens/repositories/transaction_repository.dart';
import 'package:moneylens/services/categorizer/categorizer_service.dart';
import 'package:moneylens/services/parser/parser_service.dart';
import 'package:moneylens/services/sms/sms_service.dart';
import 'package:moneylens/services/statistics/statistics_service.dart';

// ─── Service Providers (singletons, never rebuild) ───────────────────────────

final smsServiceProvider = Provider<SmsService>((_) => SmsService.instance);

final parserServiceProvider = Provider<ParserService>((_) => ParserService.instance);

final categorizerServiceProvider =
    Provider<CategorizerService>((_) => CategorizerService.instance);

final statisticsServiceProvider =
    Provider<StatisticsService>((_) => StatisticsService.instance);

// ─── Repository Providers ─────────────────────────────────────────────────────

final transactionRepositoryProvider =
    Provider<TransactionRepository>((_) => TransactionRepository.instance);

final merchantRepositoryProvider =
    Provider<MerchantRepository>((_) => MerchantRepository.instance);

final categoryRepositoryProvider =
    Provider<CategoryRepository>((_) => CategoryRepository.instance);

// ─────────────────────────────────────────────────────────────────────────────
// SMS PROVIDER
// Handles: permission check, full scan, incremental scan
// ─────────────────────────────────────────────────────────────────────────────

class SmsNotifier extends AsyncNotifier<SmsState> {
  @override
  Future<SmsState> build() async {
    final hasPermission = await ref.read(smsServiceProvider).hasPermission();
    return SmsState(
      permissionGranted: hasPermission,
      isScanning: false,
      scannedCount: 0,
      savedCount: 0,
    );
  }

  Future<bool> requestPermission() async {
    final granted = await ref.read(smsServiceProvider).requestPermission();
    state = AsyncData(state.value!.copyWith(permissionGranted: granted));
    return granted;
  }

  /// Full SMS scan → parse → categorize → save pipeline.
  Future<void> scanAll() async {
    final current = state.value ?? SmsState.empty();
    state = AsyncData(current.copyWith(isScanning: true));

    try {
      final smsResult = await ref.read(smsServiceProvider).readAllSms();

      if (!smsResult.isSuccess) {
        state = AsyncData(current.copyWith(isScanning: false));
        return;
      }

      final parsed = ref.read(parserServiceProvider).parseAll(smsResult.messages);

      final categorized = await ref
          .read(categorizerServiceProvider)
          .categorizeAll(parsed);

      final saved = await ref
          .read(transactionRepositoryProvider)
          .saveAll(categorized);

      await _saveLastScanTime();

      ref.invalidate(dashboardProvider);
      ref.invalidate(transactionListProvider);
      ref.invalidate(categoryTotalsProvider);

      state = AsyncData(current.copyWith(
        isScanning: false,
        scannedCount: smsResult.messages.length,
        savedCount: saved,
      ));
    } catch (e, st) {
      state = AsyncError(e, st);
    }
  }

  /// Incremental scan — only reads SMS since last scan.
  Future<void> scanLatest() async {
    final current = state.value ?? SmsState.empty();
    state = AsyncData(current.copyWith(isScanning: true));

    try {
      final lastScan = await _getLastScanTime();
      final smsResult =
          await ref.read(smsServiceProvider).scanLatestSms(since: lastScan);

      if (!smsResult.isSuccess || smsResult.messages.isEmpty) {
        state = AsyncData(current.copyWith(isScanning: false));
        return;
      }

      final parsed = ref.read(parserServiceProvider).parseAll(smsResult.messages);

      final categorized = await ref
          .read(categorizerServiceProvider)
          .categorizeAll(parsed);

      final saved = await ref
          .read(transactionRepositoryProvider)
          .saveAll(categorized);

      await _saveLastScanTime();

      ref.invalidate(dashboardProvider);
      ref.invalidate(transactionListProvider);
      ref.invalidate(categoryTotalsProvider);

      state = AsyncData(current.copyWith(
        isScanning: false,
        scannedCount: smsResult.messages.length,
        savedCount: saved,
      ));
    } catch (e, st) {
      state = AsyncError(e, st);
    }
  }

  Future<void> _saveLastScanTime() async {
    final db = DatabaseHelper.instance;
    await db.insert(AppConstants.tableSettings, {
      'key': AppConstants.keyLastScanTime,
      'value': DateTime.now().millisecondsSinceEpoch.toString(),
    });
  }

  Future<DateTime> _getLastScanTime() async {
    final db = DatabaseHelper.instance;
    final rows = await db.query(
      AppConstants.tableSettings,
      where: 'key = ?',
      whereArgs: [AppConstants.keyLastScanTime],
      limit: 1,
    );
    if (rows.isEmpty) {
      return DateTime.now().subtract(const Duration(days: 180));
    }
    final ms = int.tryParse(rows.first['value'] as String) ?? 0;
    return DateTime.fromMillisecondsSinceEpoch(ms);
  }
}

final smsProvider = AsyncNotifierProvider<SmsNotifier, SmsState>(SmsNotifier.new);

// ─── SMS State ────────────────────────────────────────────────────────────────

class SmsState {
  final bool permissionGranted;
  final bool isScanning;
  final int scannedCount;
  final int savedCount;

  const SmsState({
    required this.permissionGranted,
    required this.isScanning,
    required this.scannedCount,
    required this.savedCount,
  });

  static SmsState empty() => const SmsState(
        permissionGranted: false,
        isScanning: false,
        scannedCount: 0,
        savedCount: 0,
      );

  SmsState copyWith({
    bool? permissionGranted,
    bool? isScanning,
    int? scannedCount,
    int? savedCount,
  }) {
    return SmsState(
      permissionGranted: permissionGranted ?? this.permissionGranted,
      isScanning: isScanning ?? this.isScanning,
      scannedCount: scannedCount ?? this.scannedCount,
      savedCount: savedCount ?? this.savedCount,
    );
  }
}

// ─────────────────────────────────────────────────────────────────────────────
// DASHBOARD PROVIDER
// ─────────────────────────────────────────────────────────────────────────────

final dashboardProvider = FutureProvider<DashboardStats>((ref) async {
  return ref.read(statisticsServiceProvider).getDashboardStats();
});

// ─────────────────────────────────────────────────────────────────────────────
// TRANSACTION PROVIDER
// Handles: full list, search query, sort order, category filter
// ─────────────────────────────────────────────────────────────────────────────

class TransactionNotifier extends AsyncNotifier<List<TransactionModel>> {
  @override
  Future<List<TransactionModel>> build() async {
    return ref.read(transactionRepositoryProvider).getAll();
  }

  Future<void> refresh() async {
    state = const AsyncLoading();
    state = AsyncData(await ref.read(transactionRepositoryProvider).getAll());
  }

  Future<List<TransactionModel>> search(String query) async {
    if (query.trim().isEmpty) {
      return ref.read(transactionRepositoryProvider).getAll();
    }
    return ref.read(transactionRepositoryProvider).search(query.trim());
  }

  Future<List<TransactionModel>> filterByCategory(String category) async {
    return ref.read(transactionRepositoryProvider).getByCategory(category);
  }

  Future<void> deleteAll() async {
    await ref.read(transactionRepositoryProvider).deleteAll();
    state = const AsyncData([]);
    ref.invalidate(dashboardProvider);
    ref.invalidate(categoryTotalsProvider);
  }
}

final transactionListProvider =
    AsyncNotifierProvider<TransactionNotifier, List<TransactionModel>>(
  TransactionNotifier.new,
);

// ─── Search query state (drives UI search bar) ────────────────────────────────

final searchQueryProvider = StateProvider<String>((_) => '');

final filteredTransactionsProvider = FutureProvider<List<TransactionModel>>((ref) async {
  final query = ref.watch(searchQueryProvider);
  final repo = ref.read(transactionRepositoryProvider);
  if (query.trim().isEmpty) return repo.getAll();
  return repo.search(query.trim());
});

// ─── Sort order ───────────────────────────────────────────────────────────────

enum TransactionSortOrder { newestFirst, oldestFirst, highestAmount, lowestAmount }

final sortOrderProvider =
    StateProvider<TransactionSortOrder>((_) => TransactionSortOrder.newestFirst);

// ─────────────────────────────────────────────────────────────────────────────
// CATEGORY PROVIDER
// ─────────────────────────────────────────────────────────────────────────────

final categoryTotalsProvider = FutureProvider<List<CategorySummary>>((ref) async {
  return ref.read(categoryRepositoryProvider).getTotalsThisMonth();
});

/// Transactions for a specific category — used by category drill-down screen.
final categoryTransactionsProvider =
    FutureProvider.family<List<TransactionModel>, String>((ref, category) async {
  return ref.read(categoryRepositoryProvider).getTransactionsByCategory(category);
});

// ─────────────────────────────────────────────────────────────────────────────
// SETTINGS PROVIDER
// ─────────────────────────────────────────────────────────────────────────────

class SettingsNotifier extends AsyncNotifier<SettingsState> {
  @override
  Future<SettingsState> build() async {
    return _loadSettings();
  }

  Future<SettingsState> _loadSettings() async {
    final db = DatabaseHelper.instance;
    final rows = await db.query(AppConstants.tableSettings);
    final map = {
      for (final r in rows) r['key'] as String: r['value'] as String
    };

    return SettingsState(
      lastScanTime: map.containsKey(AppConstants.keyLastScanTime)
          ? DateTime.fromMillisecondsSinceEpoch(
              int.parse(map[AppConstants.keyLastScanTime]!))
          : null,
      totalScanned: int.tryParse(map[AppConstants.keyTotalScanned] ?? '0') ?? 0,
      onboardingDone: map[AppConstants.keyOnboardingDone] == 'true',
    );
  }

  Future<void> deleteAllData() async {
    await DatabaseHelper.instance.deleteAllData();
    ref.invalidate(transactionListProvider);
    ref.invalidate(dashboardProvider);
    ref.invalidate(categoryTotalsProvider);
    state = AsyncData(SettingsState.empty());
  }

  Future<void> refresh() async {
    state = AsyncData(await _loadSettings());
  }

  Future<void> markOnboardingDone() async {
    final db = DatabaseHelper.instance;
    await db.insert(AppConstants.tableSettings, {
      'key': AppConstants.keyOnboardingDone,
      'value': 'true',
    });
    state = AsyncData(
      state.value?.copyWith(onboardingDone: true) ?? SettingsState.empty(),
    );
  }
}

final settingsProvider = AsyncNotifierProvider<SettingsNotifier, SettingsState>(
  SettingsNotifier.new,
);

// ─── Settings State ───────────────────────────────────────────────────────────

class SettingsState {
  final DateTime? lastScanTime;
  final int totalScanned;
  final bool onboardingDone;

  const SettingsState({
    required this.lastScanTime,
    required this.totalScanned,
    required this.onboardingDone,
  });

  static SettingsState empty() => const SettingsState(
        lastScanTime: null,
        totalScanned: 0,
        onboardingDone: false,
      );

  SettingsState copyWith({
    DateTime? lastScanTime,
    int? totalScanned,
    int? onboardingDone,
  }) {
    return SettingsState(
      lastScanTime: lastScanTime ?? this.lastScanTime,
      totalScanned: totalScanned ?? this.totalScanned,
      onboardingDone: onboardingDone ?? this.onboardingDone,
    );
  }
}
