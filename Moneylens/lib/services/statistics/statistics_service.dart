import 'package:moneylens/models/transaction_model.dart';
import 'package:moneylens/repositories/category_repository.dart';
import 'package:moneylens/repositories/transaction_repository.dart';

class StatisticsService {
  StatisticsService._();
  static final StatisticsService instance = StatisticsService._();

  final _txRepo = TransactionRepository.instance;
  final _catRepo = CategoryRepository.instance;

  // ─── Dashboard (all-in-one) ───────────────────────────────────────────────

  /// Single call that builds the full DashboardStats object.
  /// All queries run in parallel via Future.wait for speed.
  Future<DashboardStats> getDashboardStats() async {
    final now = DateTime.now();

    // Date range boundaries
    final todayStart = DateTime(now.year, now.month, now.day);
    final todayEnd = todayStart.add(const Duration(days: 1));

    final weekStart = now.subtract(Duration(days: now.weekday - 1));
    final weekFrom = DateTime(weekStart.year, weekStart.month, weekStart.day);

    final monthStart = DateTime(now.year, now.month, 1);
    final monthEnd = DateTime(now.year, now.month + 1, 1);

    // Run all DB queries in parallel
    final results = await Future.wait([
      _txRepo.sumDebits(from: todayStart, to: todayEnd),
      _txRepo.sumDebits(from: weekFrom, to: todayEnd),
      _txRepo.sumDebits(from: monthStart, to: monthEnd),
      _catRepo.getTotalsThisMonth(),
      _txRepo.getTopMerchantThisMonth(),
      _txRepo.getRecent(limit: 10),
    ]);

    final todaySpend = results[0] as double;
    final weekSpend = results[1] as double;
    final monthSpend = results[2] as double;
    final catTotals = results[3] as List<CategorySummary>;
    final topMerchant = results[4] as String;
    final recentTxns = results[5] as List<TransactionModel>;

    final topCategory = catTotals.isNotEmpty ? catTotals.first.category : '—';

    return DashboardStats(
      todaySpend: todaySpend,
      weekSpend: weekSpend,
      monthSpend: monthSpend,
      topCategory: topCategory,
      topMerchant: topMerchant,
      categoryTotals: catTotals,
      recentTransactions: recentTxns,
    );
  }

  // ─── Individual Stat Methods ──────────────────────────────────────────────
  // Kept public so individual screens can call them without fetching everything.

  Future<double> dailySpend() async {
    final now = DateTime.now();
    final start = DateTime(now.year, now.month, now.day);
    final end = start.add(const Duration(days: 1));
    return _txRepo.sumDebits(from: start, to: end);
  }

  Future<double> weeklySpend() async {
    final now = DateTime.now();
    final ws = now.subtract(Duration(days: now.weekday - 1));
    final start = DateTime(ws.year, ws.month, ws.day);
    final end = DateTime(now.year, now.month, now.day)
        .add(const Duration(days: 1));
    return _txRepo.sumDebits(from: start, to: end);
  }

  Future<double> monthlySpend() async {
    final now = DateTime.now();
    final start = DateTime(now.year, now.month, 1);
    final end = DateTime(now.year, now.month + 1, 1);
    return _txRepo.sumDebits(from: start, to: end);
  }

  Future<List<CategorySummary>> categoryTotals() async {
    return _catRepo.getTotalsThisMonth();
  }

  Future<String> topCategory() async {
    return _catRepo.getTopCategoryThisMonth();
  }

  Future<String> topMerchant() async {
    return _txRepo.getTopMerchantThisMonth();
  }

  // ─── Spend for a Custom Range ─────────────────────────────────────────────

  Future<double> spendForRange({
    required DateTime from,
    required DateTime to,
  }) async {
    return _txRepo.sumDebits(from: from, to: to);
  }

  // ─── Daily Spend Breakdown ────────────────────────────────────────────────
  // Returns a map of { day (1–31) → total spend } for the current month.
  // Used for future sparkline / trend charts.

  Future<Map<int, double>> dailyBreakdownThisMonth() async {
    final now = DateTime.now();
    final start = DateTime(now.year, now.month, 1);
    final end = DateTime(now.year, now.month + 1, 1);

    final txns = await _txRepo.getThisMonth();
    final Map<int, double> breakdown = {};

    for (final tx in txns) {
      if (tx.isDebit &&
          tx.date.isAfter(start.subtract(const Duration(seconds: 1))) &&
          tx.date.isBefore(end)) {
        final day = tx.date.day;
        breakdown[day] = (breakdown[day] ?? 0) + tx.amount;
      }
    }

    return breakdown;
  }

  // ─── Merchant Spend Breakdown ─────────────────────────────────────────────
  // Top N merchants by spend this month.

  Future<List<MerchantSummary>> topMerchants({int limit = 5}) async {
    final txns = await _txRepo.getThisMonth();

    final Map<String, double> totals = {};
    for (final tx in txns) {
      if (tx.isDebit) {
        totals[tx.merchant] = (totals[tx.merchant] ?? 0) + tx.amount;
      }
    }

    final sorted = totals.entries.toList()
      ..sort((a, b) => b.value.compareTo(a.value));

    return sorted
        .take(limit)
        .map((e) => MerchantSummary(merchant: e.key, totalAmount: e.value))
        .toList();
  }

  // ─── Transaction Count Helpers ────────────────────────────────────────────

  Future<int> totalTransactionCount() async {
    final all = await _txRepo.getAll();
    return all.length;
  }

  Future<int> transactionCountThisMonth() async {
    final txns = await _txRepo.getThisMonth();
    return txns.length;
  }
}

// ─── Supporting Model ─────────────────────────────────────────────────────────

class MerchantSummary {
  final String merchant;
  final double totalAmount;

  const MerchantSummary({
    required this.merchant,
    required this.totalAmount,
  });
}
